from src.vision.damage_detector import DamageDetector


model_path = "runs/detect/runs/damage/autoclaim_detector_test/weights/best.pt"

image_path = "data/raw/test/images/signlight-10-53-_jpg.rf.8769dd83f6c826b27b7c164dca5d50c4.jpg"


detector = DamageDetector(model_path)

detections = detector.predict(
    image_path,
    confidence=0.25,
    
)


print("\n================================")
print("       AUTOCLAIM AI RESULT")
print("================================")

if not detections:

    print("No damage detected.")

else:

    print(f"Damages detected: {len(detections)}\n")

    for i, detection in enumerate(detections, start=1):

        print(f"Damage {i}")
        print("----------------------------")
        print(f"Type       : {detection['damage_type']}")
        print(f"Confidence : {detection['confidence'] * 100:.1f}%")
        print(
            f"Bounding Box: "
            f"{[round(x, 2) for x in detection['box_xyxy']]}"
        )
        print()