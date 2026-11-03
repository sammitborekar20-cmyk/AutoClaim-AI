from ultralytics import YOLO


class DamageDetector:

    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def predict(self, image_path, confidence=0.25):

        # Run YOLO prediction
        results = self.model.predict(
            source=image_path,
            conf=confidence
        )

        detections = []

        # Read all detections
        for result in results:

            for box in result.boxes:

                class_id = int(box.cls[0])
                confidence_score = float(box.conf[0])

                bounding_box = box.xyxy[0].tolist()

                detections.append({
                    "class_id": class_id,
                    "damage_type": self.model.names[class_id],
                    "confidence": confidence_score,
                    "box_xyxy": bounding_box
                })

        # Sort by confidence
        detections.sort(
            key=lambda x: x["confidence"],
            reverse=True
        )

        # Keep only the strongest prediction
        if detections:
            return [detections[0]]

        return []