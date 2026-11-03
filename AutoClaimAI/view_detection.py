from ultralytics import YOLO


model_path = "runs/detect/runs/damage/autoclaim_detector_test/weights/best.pt"

image_path = "data/raw/test/images/signlight-10-53-_jpg.rf.8769dd83f6c826b27b7c164dca5d50c4.jpg"


model = YOLO(model_path)

results = model.predict(
    source=image_path,
    conf=0.25,
    save=True
)

print("\nDetection completed!")

print("The annotated image has been saved by YOLO.")