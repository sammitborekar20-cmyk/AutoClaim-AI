from pathlib import Path
from datetime import date
import csv
import shutil
import subprocess

import cv2


# ============================================================
# PATHS
# ============================================================

IMAGE_DIR = Path("data/raw/train/images")
LABEL_DIR = Path("data/raw/train/labels")

CLEANED_IMAGE_DIR = Path("data/cleaned/images")
CLEANED_LABEL_DIR = Path("data/cleaned/labels")

CHANGE_LOG = Path("reports/annotation_changes.csv")
REVIEW_LOG = Path("reports/reviewed_images.csv")

REVIEW_IMAGE = Path("data/cleaned/review_current.jpg")

# Number of damage classes in the dataset
NUM_CLASSES = 14


# ============================================================
# CLASS NAMES
# Keep the order consistent with data.yaml
# ============================================================

CLASS_NAMES = [
    "bonnet-dent",
    "boot-dent",
    "doorouter-dent",
    "fender-dent",
    "front-bumper-dent",
    "Front-windscreen-damage",
    "Headlight-damage",
    "quaterpanel-dent",
    "rear-bumper-dent",
    "Rear-windscreen-Damage",
    "roof-dent",
    "Runningboard-Damage",
    "Sidemirror-Damage",
    "Taillight-Damage",
]


# ============================================================
# CSV HEADERS
# ============================================================

REVIEW_COLUMNS = [
    "image",
    "split",
    "review_status",
    "review_date",
    "notes",
]

CHANGE_COLUMNS = [
    "image",
    "split",
    "action",
    "old_class",
    "new_class",
    "old_box",
    "new_box",
    "reason",
    "reviewer",
    "status",
]


# ============================================================
# SETUP
# ============================================================

for directory in [
    CLEANED_IMAGE_DIR,
    CLEANED_LABEL_DIR,
    REVIEW_IMAGE.parent,
    REVIEW_LOG.parent,
    CHANGE_LOG.parent,
]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# GET TRAINING IMAGES
# ============================================================

if not IMAGE_DIR.exists():
    raise FileNotFoundError(
        f"Training image directory not found: {IMAGE_DIR}"
    )

image_files = sorted(
    path
    for path in IMAGE_DIR.iterdir()
    if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
)


# ============================================================
# READ YOLO LABELS
#
# YOLO format:
# class_id x_center y_center width height
#
# Coordinates are normalized between 0 and 1.
# ============================================================

def read_labels(label_path):
    """
    Return valid YOLO annotations as a list of strings.

    Invalid rows are reported and excluded from the returned list.
    The source file is never modified by this function.
    """

    if not label_path.exists():
        return []

    annotations = []

    with open(label_path, "r") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            values = line.split()

            if len(values) != 5:
                print(
                    f"WARNING: {label_path.name}, line "
                    f"{line_number}: expected 5 values."
                )
                continue

            try:
                class_id = int(values[0])
                x_center, y_center, box_width, box_height = map(
                    float, values[1:]
                )
            except ValueError:
                print(
                    f"WARNING: Invalid numbers in "
                    f"{label_path.name}, line {line_number}."
                )
                continue

            if not 0 <= class_id < NUM_CLASSES:
                print(
                    f"WARNING: Invalid class ID {class_id} "
                    f"in {label_path.name}, line {line_number}."
                )
                continue

            if not all(
                0 <= value <= 1
                for value in [
                    x_center,
                    y_center,
                    box_width,
                    box_height,
                ]
            ):
                print(
                    f"WARNING: Coordinates outside [0, 1] "
                    f"in {label_path.name}, line {line_number}."
                )
                continue

            if box_width <= 0 or box_height <= 0:
                print(
                    f"WARNING: Zero or negative box size "
                    f"in {label_path.name}, line {line_number}."
                )
                continue

            # Store valid labels in standard YOLO format.
            annotations.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} "
                f"{box_width:.6f} {box_height:.6f}"
            )

    return annotations


# ============================================================
# WRITE CLEANED LABELS
# ============================================================

def write_labels(label_path, annotations):
    """
    Save annotations only in the cleaned directory.
    Never write to data/raw/train/labels/.
    """

    label_path.parent.mkdir(parents=True, exist_ok=True)

    with open(label_path, "w") as file:
        for annotation in annotations:
            file.write(annotation.strip() + "\n")


# ============================================================
# READ REVIEW LOG
# ============================================================

def read_review_log():
    """
    Read the review log and keep one record per image.
    Existing records are preserved.
    """

    records = {}

    if not REVIEW_LOG.exists():
        return records

    try:
        with open(REVIEW_LOG, "r", newline="") as file:
            reader = csv.DictReader(file)

            for row in reader:
                image_name = (row.get("image") or "").strip()

                if not image_name:
                    continue

                records[image_name] = {
                    "image": image_name,
                    "split": row.get("split", "train"),
                    "review_status": row.get(
                        "review_status", "UNREVIEWED"
                    ),
                    "review_date": row.get("review_date", ""),
                    "notes": row.get("notes", ""),
                }

    except (OSError, csv.Error) as error:
        raise RuntimeError(
            f"Could not safely read {REVIEW_LOG}: {error}"
        ) from error

    return records


# ============================================================
# SAVE REVIEW LOG
# ============================================================

def save_review_log(records):
    """
    Rewrite the review log with one row per image.
    """

    with open(REVIEW_LOG, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=REVIEW_COLUMNS,
        )
        writer.writeheader()

        for image_name in sorted(records):
            writer.writerow(records[image_name])


# ============================================================
# RECORD REVIEW STATUS
# ============================================================

def log_review(image_path, status, notes):
    records = read_review_log()

    records[image_path.name] = {
        "image": image_path.name,
        "split": "train",
        "review_status": status,
        "review_date": date.today().isoformat(),
        "notes": notes,
    }

    save_review_log(records)


# ============================================================
# LOG ANNOTATION CHANGE
# ============================================================

def log_change(
    image_path,
    action,
    old_class="",
    new_class="",
    old_box="",
    new_box="",
    reason="",
    status="SAVED",
):
    """
    Append a change record. Existing change history is preserved.
    """

    needs_header = (
        not CHANGE_LOG.exists()
        or CHANGE_LOG.stat().st_size == 0
    )

    with open(CHANGE_LOG, "a", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=CHANGE_COLUMNS,
        )

        if needs_header:
            writer.writeheader()

        writer.writerow({
            "image": image_path.name,
            "split": "train",
            "action": action,
            "old_class": old_class,
            "new_class": new_class,
            "old_box": old_box,
            "new_box": new_box,
            "reason": reason,
            "reviewer": "manual",
            "status": status,
        })


# ============================================================
# COPY ORIGINAL IMAGE TO CLEANED DIRECTORY
# ============================================================

def copy_image(image_path):
    destination = CLEANED_IMAGE_DIR / image_path.name

    if not destination.exists():
        shutil.copy2(image_path, destination)

    return destination


# ============================================================
# GET LABEL PATH USED FOR REVIEW
#
# Prefer cleaned labels if available.
# Otherwise, use original labels for viewing.
# ============================================================

def get_working_labels(image_path):
    original_label = LABEL_DIR / f"{image_path.stem}.txt"
    cleaned_label = CLEANED_LABEL_DIR / f"{image_path.stem}.txt"

    if cleaned_label.exists():
        return cleaned_label, read_labels(cleaned_label)

    return original_label, read_labels(original_label)


# ============================================================
# DISPLAY ANNOTATIONS
# ============================================================

def display_annotation(image_path, annotations):
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"ERROR: Could not load {image_path}")
        return False

    height, width = image.shape[:2]

    print()
    print("=" * 60)
    print(f"IMAGE: {image_path.name}")
    print(f"Image dimensions: {width} x {height}")
    print(f"Annotations: {len(annotations)}")
    print("=" * 60)

    for index, annotation in enumerate(annotations):
        values = annotation.split()

        class_id = int(values[0])
        x_center, y_center, box_width, box_height = map(
            float, values[1:]
        )

        x1 = int((x_center - box_width / 2) * width)
        y1 = int((y_center - box_height / 2) * height)
        x2 = int((x_center + box_width / 2) * width)
        y2 = int((y_center + box_height / 2) * height)

        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))

        class_name = CLASS_NAMES[class_id]

        print(
            f"Box {index + 1}: class {class_id} "
            f"({class_name}) | "
            f"YOLO box: {values[1:]}"
        )

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            image,
            f"{index + 1}: {class_name}",
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
        )

    success = cv2.imwrite(str(REVIEW_IMAGE), image)

    if not success:
        print("WARNING: Could not save the preview image.")
        return False

    print(f"\nPreview saved at: {REVIEW_IMAGE}")

    # Try terminal preview if chafa is installed.
    try:
        result = subprocess.run(
            [
                "chafa",
                "--format", "symbols",
                "--size", "80x30",
                str(REVIEW_IMAGE),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0 and result.stdout.strip():
            print(result.stdout)
            return True

    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # On macOS, open the preview in Preview.app.
    print("Opening annotated image for visual inspection...")

    try:
        subprocess.Popen(["open", str(REVIEW_IMAGE)])
    except OSError:
        print(f"Open this image manually: {REVIEW_IMAGE.resolve()}")

    return True


# ============================================================
# VALIDATE ONE BOX
# ============================================================

def validate_box(class_id, x_center, y_center, box_width, box_height):
    if not 0 <= class_id < NUM_CLASSES:
        return False, f"Class ID must be between 0 and {NUM_CLASSES - 1}."

    if not all(
        0 <= value <= 1
        for value in [
            x_center,
            y_center,
            box_width,
            box_height,
        ]
    ):
        return False, "All normalized coordinates must be between 0 and 1."

    if box_width <= 0 or box_height <= 0:
        return False, "Box width and height must be greater than zero."

    if x_center - box_width / 2 < 0:
        return False, "Box extends beyond the left edge."

    if y_center - box_height / 2 < 0:
        return False, "Box extends beyond the top edge."

    if x_center + box_width / 2 > 1:
        return False, "Box extends beyond the right edge."

    if y_center + box_height / 2 > 1:
        return False, "Box extends beyond the bottom edge."

    return True, "Valid"


# ============================================================
# CORRECT AN ANNOTATION
#
# Enter normalized YOLO coordinates:
# x_center y_center width height
# ============================================================

def correct_annotation(image_path, annotations):
    if not annotations:
        print("No annotation boxes to correct.")
        return annotations

    print("\nChoose the box number to correct:")
    choice = input("Box number (or Enter to cancel): ").strip()

    if not choice:
        return annotations

    try:
        index = int(choice) - 1
    except ValueError:
        print("Invalid box number.")
        return annotations

    if not 0 <= index < len(annotations):
        print("Box number does not exist.")
        return annotations

    old_values = annotations[index].split()
    old_class = int(old_values[0])
    old_box = " ".join(old_values[1:])

    print("\nCurrent annotation:")
    print(f"Class: {old_class} ({CLASS_NAMES[old_class]})")
    print(f"YOLO coordinates: {old_box}")

    print("\nAvailable classes:")
    for class_id, class_name in enumerate(CLASS_NAMES):
        print(f"  {class_id}: {class_name}")

    class_input = input(
        f"\nNew class ID [Enter to keep {old_class}]: "
    ).strip()

    try:
        new_class = int(class_input) if class_input else old_class
    except ValueError:
        print("Invalid class ID. No changes made.")
        return annotations

    coord_input = input(
        "New x_center y_center width height "
        "[Enter to keep current coordinates]: "
    ).strip()

    if coord_input:
        try:
            coords = list(map(float, coord_input.split()))

            if len(coords) != 4:
                raise ValueError

            x_center, y_center, box_width, box_height = coords

        except ValueError:
            print("Enter exactly four numeric coordinates. No changes made.")
            return annotations

    else:
        x_center, y_center, box_width, box_height = map(
            float, old_values[1:]
        )

    valid, message = validate_box(
        new_class,
        x_center,
        y_center,
        box_width,
        box_height,
    )

    if not valid:
        print(f"Correction rejected: {message}")
        return annotations

    new_box = (
        f"{x_center:.6f} {y_center:.6f} "
        f"{box_width:.6f} {box_height:.6f}"
    )

    annotations[index] = f"{new_class} {new_box}"

    # Save changes in cleaned labels only.
    cleaned_label = CLEANED_LABEL_DIR / f"{image_path.stem}.txt"
    write_labels(cleaned_label, annotations)
    copy_image(image_path)

    log_change(
        image_path,
        action="CORRECT",
        old_class=old_class,
        new_class=new_class,
        old_box=old_box,
        new_box=new_box,
        reason="Manual correction after visual inspection.",
    )

    print("Correction saved in the cleaned dataset.")
    return annotations


# ============================================================
# REMOVE AN ANNOTATION
# ============================================================

def remove_annotation(image_path, annotations):
    if not annotations:
        print("No annotation boxes to remove.")
        return annotations

    print("\nChoose the box number to remove:")
    choice = input("Box number (or Enter to cancel): ").strip()

    if not choice:
        return annotations

    try:
        index = int(choice) - 1
    except ValueError:
        print("Invalid box number.")
        return annotations

    if not 0 <= index < len(annotations):
        print("Box number does not exist.")
        return annotations

    old_values = annotations[index].split()
    old_class = int(old_values[0])
    old_box = " ".join(old_values[1:])

    print(
        f"Selected: {CLASS_NAMES[old_class]} | "
        f"{old_box}"
    )

    reason = input(
        "Why remove this box? "
        "(e.g. incorrect label, no visible damage): "
    ).strip()

    if not reason:
        reason = "Removed after manual visual inspection."

    confirm = input(
        "Confirm removal? [y/N]: "
    ).strip().lower()

    if confirm != "y":
        print("Removal cancelled.")
        return annotations

    annotations.pop(index)

    cleaned_label = CLEANED_LABEL_DIR / f"{image_path.stem}.txt"
    write_labels(cleaned_label, annotations)
    copy_image(image_path)

    log_change(
        image_path,
        action="REMOVE",
        old_class=old_class,
        new_class="",
        old_box=old_box,
        new_box="",
        reason=reason,
    )

    print("Box removed from the cleaned dataset.")
    return annotations


# ============================================================
# KEEP ANNOTATIONS UNCHANGED
# ============================================================

def keep_annotations(image_path, annotations):
    """
    Copy the original labels to cleaned labels only if a cleaned
    label file does not already exist. This preserves corrections.
    """

    cleaned_label = CLEANED_LABEL_DIR / f"{image_path.stem}.txt"
    original_label = LABEL_DIR / f"{image_path.stem}.txt"

    copy_image(image_path)

    if not cleaned_label.exists():
        if original_label.exists():
            shutil.copy2(original_label, cleaned_label)
        else:
            write_labels(cleaned_label, annotations)

    log_review(
        image_path,
        "REVIEWED",
        "Visually reviewed and accepted.",
    )

    print("Image marked as REVIEWED.")


# ============================================================
# MAIN REVIEW LOOP
# ============================================================

def main():
    if not image_files:
        print(f"No images found in {IMAGE_DIR}")
        return

    print("\n" + "=" * 60)
    print("        AUTOCLAIM AI ANNOTATION REVIEW")
    print("=" * 60)

    print(f"Training images available: {len(image_files)}")

    review_records = read_review_log()

    reviewed = {
        name
        for name, record in review_records.items()
        if record.get("review_status") == "REVIEWED"
    }

    pending_images = [
        image
        for image in image_files
        if image.name not in reviewed
    ]

    print(f"Already reviewed: {len(reviewed)}")
    print(f"Images remaining: {len(pending_images)}")

    print("\nControls:")
    print("  k = Keep annotations and mark reviewed")
    print("  c = Correct a box class/coordinates")
    print("  r = Remove a box")
    print("  s = Skip image")
    print("  q = Quit")

    for image_number, image_path in enumerate(
        pending_images, start=1
    ):
        print("\n" + "-" * 60)
        print(
            f"Pending image {image_number}/{len(pending_images)}"
        )
        print(f"Image: {image_path.name}")

        _, annotations = get_working_labels(image_path)

        while True:
            display_annotation(image_path, annotations)

            print("\nChoose: k / c / r / s / q")
            choice = input("Decision: ").strip().lower()

            if choice == "k":
                keep_annotations(image_path, annotations)
                break

            elif choice == "c":
                annotations = correct_annotation(
                    image_path, annotations
                )

            elif choice == "r":
                annotations = remove_annotation(
                    image_path, annotations
                )

            elif choice == "s":
                log_review(
                    image_path,
                    "UNREVIEWED",
                    "Skipped for later review.",
                )
                print("Skipped; it will remain pending.")
                break

            elif choice == "q":
                print("\nReview stopped.")
                print(f"Review log: {REVIEW_LOG}")
                print(f"Change log: {CHANGE_LOG}")
                return

            else:
                print("Invalid choice. Enter k, c, r, s, or q.")

    print("\n" + "=" * 60)
    print("        REVIEW SESSION COMPLETE")
    print("=" * 60)
    print(f"Review log: {REVIEW_LOG}")
    print(f"Change log: {CHANGE_LOG}")
    print(f"Cleaned images: {CLEANED_IMAGE_DIR}")
    print(f"Cleaned labels: {CLEANED_LABEL_DIR}")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()