from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path = "models/yolo11n.pt", confidence = 0.5):
        try:    
            self.model = YOLO(model_path)
            self.confidence = confidence

        except Exception as e:
            print(e)

    def detect(self, frame):
        try:
            results = self.model.predict(source = frame, conf = self.confidence, verbose = False)
            if not results:
                return None
            
            return results[0]

        except Exception as e:
            print(e)

    def draw_detections(self, frame):
        try:
            result = self.detect(frame)
            if result is None:
                return frame, None

            annotated_frame = result.plot()

            return annotated_frame, result

        except Exception as e:
            print(e)

    def get_detection_data(self, result):
        try:
            detections = []
            if result is None:
                return detections

            if result.boxes is None:
                return detections

            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = result.names[class_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                detections.append({
                    "class": class_name,
                    "confidence": confidence,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                })

            return detections

        except Exception as e:
            print(e)