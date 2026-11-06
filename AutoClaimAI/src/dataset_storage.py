from pathlib import Path

# --------------------------------------------------
# AUTOCLAIM AI - DATASET STORAGE SIZE
# Calculates total dataset storage in MB and GB
# --------------------------------------------------

DATASET_PATH = Path("data/raw")


def main():
    if not DATASET_PATH.exists():
        print("ERROR: Dataset folder was not found.")
        print(f"Checked path: {DATASET_PATH.resolve()}")
        return

    total_bytes = sum(
        file.stat().st_size
        for file in DATASET_PATH.rglob("*")
        if file.is_file()
    )

    size_mb = total_bytes / (1024 ** 2)
    size_gb = total_bytes / (1024 ** 3)

    print("\n==========================================")
    print("       AUTOCLAIM AI DATASET STORAGE")
    print("==========================================")
    print(f"Dataset folder: {DATASET_PATH.resolve()}")
    print(f"Storage size:   {size_mb:.2f} MB")
    print(f"Storage size:   {size_gb:.2f} GB")
    print("==========================================\n")


if __name__ == "__main__":
    main()