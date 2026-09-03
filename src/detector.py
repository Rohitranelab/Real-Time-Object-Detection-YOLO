from ultralytics import YOLO


class ObjectDetector:

    def __init__(self, model_path="models/yolo11n.pt", confidence=0.5):

        self.model = YOLO(model_path)
        self.confidence = confidence

    def detect(self, frame):

        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            verbose=False
        )

        if not results:
            return None

        # YOLO returns a list.
        # Return the first Results object.
        return results[0]

    def draw_detections(self, frame):

        result = self.detect(frame)

        if result is None:
            return frame, None

        # result is now a YOLO Results object
        annotated_frame = result.plot()

        return annotated_frame, result

    def get_detection_data(self, result):

        detections = []

        if result is None:
            return detections

        if result.boxes is None:
            return detections

        for box in result.boxes:

            class_id = int(box.cls[0])

            confidence = float(box.conf[0])

            class_name = result.names[class_id]

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            detections.append({
                "class": class_name,
                "confidence": confidence,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            })

        return detections