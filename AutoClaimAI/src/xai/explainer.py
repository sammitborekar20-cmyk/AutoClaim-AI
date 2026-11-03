import numpy as np
import cv2
from PIL import Image


def generate_occlusion_explanation(
    model,
    image,
    target_class_id=None,
    grid_size=3,
):
    """
    Generate a perturbation-based explanation for one YOLO detection.

    Parameters
    ----------
    model:
        Loaded Ultralytics YOLO model.
    image:
        PIL Image, preferably RGB.
    target_class_id:
        Class ID to explain. If None, use the highest-confidence detection.
    grid_size:
        Number of rows and columns of image patches to test.

    Returns
    -------
    dict with baseline confidence, target class, heatmap, overlay, and message.

    Note:
        This performs additional model inferences and may take some time.
    """

    if grid_size < 2 or grid_size > 5:
        raise ValueError("grid_size must be between 2 and 5.")

    original = np.asarray(image.convert("RGB")).copy()
    height, width = original.shape[:2]

    baseline_result = model.predict(
        source=image,
        conf=0.01,
        verbose=False,
    )[0]

    if baseline_result.boxes is None or len(baseline_result.boxes) == 0:
        return {
            "success": False,
            "message": "No detection is available to explain.",
        }

    classes = baseline_result.boxes.cls.cpu().numpy().astype(int)
    confidences = baseline_result.boxes.conf.cpu().numpy()

    if target_class_id is None:
        best_index = int(np.argmax(confidences))
        target_class_id = int(classes[best_index])
        baseline_confidence = float(confidences[best_index])
    else:
        matching = np.where(classes == int(target_class_id))[0]

        if len(matching) == 0:
            return {
                "success": False,
                "message": (
                    f"Class {target_class_id} was not detected in the image."
                ),
            }

        best_index = int(matching[np.argmax(confidences[matching])])
        baseline_confidence = float(confidences[best_index])

    class_name = baseline_result.names.get(
        int(target_class_id), f"class_{target_class_id}"
    )

    # Each cell stores how much the selected class's confidence drops
    # when that image region is covered.
    importance = np.zeros((grid_size, grid_size), dtype=np.float32)

    for row in range(grid_size):
        y1 = round(row * height / grid_size)
        y2 = round((row + 1) * height / grid_size)

        for col in range(grid_size):
            x1 = round(col * width / grid_size)
            x2 = round((col + 1) * width / grid_size)

            masked = original.copy()

            # Replace this patch with its local mean colour.
            patch = masked[y1:y2, x1:x2]
            mean_colour = patch.mean(axis=(0, 1)).astype(np.uint8)
            masked[y1:y2, x1:x2] = mean_colour

            masked_image = Image.fromarray(masked)

            result = model.predict(
                source=masked_image,
                conf=0.01,
                verbose=False,
            )[0]

            remaining_confidence = 0.0

            if result.boxes is not None and len(result.boxes) > 0:
                result_classes = (
                    result.boxes.cls.cpu().numpy().astype(int)
                )
                result_confidences = result.boxes.conf.cpu().numpy()

                same_class = result_confidences[
                    result_classes == int(target_class_id)
                ]

                if len(same_class) > 0:
                    remaining_confidence = float(np.max(same_class))

            importance[row, col] = max(
                0.0, baseline_confidence - remaining_confidence
            )

    # Expand the coarse grid into a smooth image-sized heatmap.
    heatmap = cv2.resize(
        importance,
        (width, height),
        interpolation=cv2.INTER_CUBIC,
    )

    heatmap = np.maximum(heatmap, 0)

    maximum = float(heatmap.max())
    if maximum > 0:
        heatmap = heatmap / maximum
    else:
        heatmap = np.zeros_like(heatmap)

    heatmap_uint8 = np.uint8(255 * heatmap)
    coloured_heatmap = cv2.applyColorMap(
        heatmap_uint8, cv2.COLORMAP_JET
    )
    coloured_heatmap = cv2.cvtColor(
        coloured_heatmap, cv2.COLOR_BGR2RGB
    )

    overlay = cv2.addWeighted(
        original, 0.55, coloured_heatmap, 0.45, 0
    )

    return {
        "success": True,
        "class_id": int(target_class_id),
        "class_name": class_name,
        "baseline_confidence": baseline_confidence,
        "heatmap": coloured_heatmap,
        "overlay": overlay,
        "importance_grid": importance.tolist(),
        "message": (
            "Redder regions indicate areas where masking caused a larger "
            "drop in the selected class's confidence. This is an approximate "
            "occlusion explanation, not proof of causal reasoning."
        ),
    }