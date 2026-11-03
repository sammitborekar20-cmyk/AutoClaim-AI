from ultralytics import YOLO

MODEL_PATH = (
    "runs/detect/runs/damage/"
    "autoclaim_20epochs/weights/best.pt"
)

DATA_PATH = "data/raw/data.yaml"


def main():
    print()
    print("========================================")
    print("   AUTOCLAIM AI - VISION EVALUATION")
    print("========================================")
    print()

    print("Loading 20-epoch YOLO model...")
    model = YOLO(MODEL_PATH)

    print("Evaluating on unseen TEST dataset...")
    print()

    results = model.val(
        data=DATA_PATH,
        split="test",
        device="mps",
        plots=True,
        project="runs/damage",
        name="test_evaluation_20epochs",
        exist_ok=True
    )

    print()
    print("========================================")
    print("       TEST EVALUATION COMPLETED")
    print("========================================")
    print()

    print(f"Precision : {results.box.mp:.3f}")
    print(f"Recall    : {results.box.mr:.3f}")
    print(f"mAP50     : {results.box.map50:.3f}")
    print(f"mAP50-95  : {results.box.map:.3f}")

    print()
    print("Results saved in:")
    print(
        "runs/detect/runs/damage/"
        "test_evaluation_20epochs"
    )


if __name__ == "__main__":
    main()