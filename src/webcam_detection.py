import cv2

def run_webcam(detector):
    try:
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Could not open WebCam")
            return

        while True:
            ret, frame = cap.read()

            if not ret:
                print("Error: Could not read frame")
                break

            annotated_frame, _ = detector.draw_detections(frame)
            cv2.imshow("YOLO11 Real-Time Object Detection", annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    except Exception as e:
        print(e)