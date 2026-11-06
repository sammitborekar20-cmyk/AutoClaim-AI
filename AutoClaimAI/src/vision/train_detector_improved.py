
from ultralytics import YOLO
from pathlib import Path

# ==========================================
# AUTOCLAIM AI - IMPROVED YOLO11 TRAINING
# ==========================================

# Start from your best existing trained model
MODEL_PATH = (
    "runs/detect/runs/damage/"
    "autoclaim_20epochs/weights/best.pt"
)

DATA_YAML = "data/raw/data.yaml"

if not Path(MODEL_PATH).exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not Path(DATA_YAML).exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_YAML}")

print("Loading existing best YOLO11n model...")
model = YOLO(MODEL_PATH)

print("Starting improved training on Apple M1 MPS...")

results = model.train(
    data=DATA_YAML,

    # Training
    epochs=20,
    imgsz=640,
    batch=8,
    device="mps",
    patience=10,

    # Fine-tuning learning rate
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,
    weight_decay=0.0005,
    cos_lr=True,

    # Moderate image augmentation
    hsv_h=0.015,
    hsv_s=0.5,
    hsv_v=0.3,
    degrees=3.0,
    translate=0.1,
    scale=0.4,
    fliplr=0.5,
    flipud=0.0,
    mosaic=0.8,
    mixup=0.0,
    close_mosaic=5,

    # Save as a separate experiment
    project="runs/detect/runs/damage",
    name="autoclaim_improved_v1",
    exist_ok=False,

    plots=True,
    save=True,
    workers=4,
    seed=42
)

print("\n======================================")
print("IMPROVED TRAINING COMPLETED")
print("======================================")
print("Best model:")
print(
    "runs/detect/runs/damage/"
    "autoclaim_improved_v1/weights/best.pt"
)
print("======================================")
