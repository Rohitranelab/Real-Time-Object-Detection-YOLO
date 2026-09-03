import cv2
import numpy as np


def process_image(image, detector):

    image_array = np.array(image)

    image_bgr = cv2.cvtColor(
        image_array,
        cv2.COLOR_RGB2BGR
    )

    annotated_frame, result = detector.draw_detections(
        image_bgr
    )

    detections = detector.get_detection_data(
        result
    )

    annotated_frame = cv2.cvtColor(
        annotated_frame,
        cv2.COLOR_BGR2RGB
    )

    return annotated_frame, detections