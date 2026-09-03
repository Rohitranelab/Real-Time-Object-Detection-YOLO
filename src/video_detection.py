import cv2
import tempfile


def process_video(video_path, detector):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Unable to open video file.")

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25

    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    )

    output_path = temp_file.name
    temp_file.close()

    # IMPORTANT: exactly 4 characters
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise ValueError(
            "Unable to create output video."
        )

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        annotated_frame, result = detector.draw_detections(
            frame
        )

        writer.write(annotated_frame)

    cap.release()
    writer.release()

    return output_path