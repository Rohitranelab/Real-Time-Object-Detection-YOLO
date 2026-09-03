import streamlit as st
from PIL import Image
import os

from src.detector import ObjectDetector
from src.image_detection import process_image
from src.video_detection import process_video
from src.webcam_detection import run_webcam
from src.utils import (
    count_objects,
    total_objects,
    average_confidence
)


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="YOLO11 Object Detection",
    page_icon="🎯",
    layout="wide"
)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🎯 Real-Time Object Detection System")

st.write(
    "Object detection using YOLO11, Python and OpenCV"
)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.header("Detection Settings")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.5,
    step=0.05
)

mode = st.sidebar.selectbox(
    "Select Detection Mode",
    [
        "Image Detection",
        "Video Detection",
        "Webcam Detection"
    ]
)


# --------------------------------------------------
# Load Model
# --------------------------------------------------

@st.cache_resource
def load_detector(confidence):

    return ObjectDetector(
        model_path="models/yolo11n.pt",
        confidence=confidence
    )


detector = load_detector(confidence)


# --------------------------------------------------
# IMAGE DETECTION
# --------------------------------------------------

if mode == "Image Detection":

    st.header("🖼️ Image Object Detection")

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

    if uploaded_file:

        image = Image.open(
            uploaded_file
        )

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Original Image")

            st.image(
                image,
                use_container_width=True
            )

        annotated_image, detections = process_image(
            image,
            detector
        )

        with col2:

            st.subheader("Detected Objects")

            st.image(
                annotated_image,
                use_container_width=True
            )

        # Statistics

        st.divider()

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Objects Detected",
                total_objects(detections)
            )

        with col2:

            st.metric(
                "Average Confidence",
                f"{average_confidence(detections):.2%}"
            )

        with col3:

            st.metric(
                "Object Classes",
                len(count_objects(detections))
            )

        # Object count

        st.subheader("📊 Object Count")

        object_counts = count_objects(
            detections
        )

        if object_counts:

            st.bar_chart(
                object_counts
            )

        else:

            st.info(
                "No objects detected."
            )


# --------------------------------------------------
# VIDEO DETECTION
# --------------------------------------------------

elif mode == "Video Detection":

    st.header("🎥 Video Object Detection")

    uploaded_video = st.file_uploader(
        "Upload a video",
        type=[
            "mp4",
            "avi",
            "mov",
            "mkv"
        ]
    )

    if uploaded_video:

        temp_input = "input_video.mp4"

        with open(
            temp_input,
            "wb"
        ) as f:

            f.write(
                uploaded_video.read()
            )

        st.info(
            "Processing video..."
        )

        try:

            output_path = process_video(
                temp_input,
                detector
            )

            st.success(
                "Video processing completed!"
            )

            with open(
                output_path,
                "rb"
            ) as f:

                video_bytes = f.read()

            st.video(video_bytes)

            st.download_button(
                label="⬇️ Download Processed Video",
                data=video_bytes,
                file_name="detected_video.mp4",
                mime="video/mp4"
            )

        except Exception as e:

            st.error(
                f"Error: {e}"
            )


# --------------------------------------------------
# WEBCAM DETECTION
# --------------------------------------------------

elif mode == "Webcam Detection":

    st.header("📷 Real-Time Webcam Detection")

    st.warning(
        "The webcam window will open locally. "
        "Press 'q' to stop detection."
    )

    if st.button(
        "Start Webcam Detection"
    ):

        run_webcam(
            detector
        )