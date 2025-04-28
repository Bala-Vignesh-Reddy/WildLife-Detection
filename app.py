import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from datetime import datetime
from ultralytics import YOLO
import pandas as pd

model = YOLO("final.pt")

def camera_detection(confidence):
    st.subheader("Live Camera Detection")
    frame_placeholder = st.empty()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not open camera.")
        return
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to capture image from camera")
        cap.release()
        return
    results = process_frame(frame, confidence)
    frame_placeholder.image(results["image"], channels="BGR", use_container_width=True)
    if results["detections"]:
        record_detection(results["detections"], "Camera")
    cap.release()

def image_detection(confidence):
    st.subheader("Image Detection")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.image(image, caption="Uploaded Image", channels="BGR", use_container_width=True)
        if st.button("Detect Animals", key=1):
            results = process_frame(image, confidence)
            st.image(results["image"], caption="Detection Result", channels="BGR", use_container_width=True)
            if results["detections"]:
                record_detection(results["detections"], f"Image: {uploaded_file.name}")

def video_detection(confidence):
    st.subheader("Video Detection")
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        frame_placeholder = st.empty()
        cap = cv2.VideoCapture(tfile.name)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = process_frame(frame, confidence)
            frame_placeholder.image(results["image"], channels="BGR", use_container_width=True)
            if results["detections"]:
                record_detection(results["detections"], f"Video: {uploaded_file.name}")
        cap.release()
        os.unlink(tfile.name)

def process_frame(frame, confidence):
    results = model(frame, conf=confidence)[0]
    detections = []
    annotated_frame = frame.copy()
    class_names = ["elephant", "leopard", "lion", "monkey"]
    if len(results.boxes) > 0:
        for box in results.boxes:
            class_id = int(box.cls[0].item())
            conf = box.conf[0].item()
            bbox = box.xyxy[0].cpu().numpy().astype(int)
            if conf >= confidence:
                detections.append({
                    "class": class_names[class_id],
                    "confidence": conf,
                    "bbox": bbox,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                cv2.rectangle(
                    annotated_frame,
                    (bbox[0], bbox[1]),
                    (bbox[2], bbox[3]),
                    (0, 255, 0),
                    2
                )
                label = f"{class_names[class_id]}: {conf:.2f}"
                cv2.putText(
                    annotated_frame,
                    label,
                    (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )
    return {
        "image": annotated_frame,
        "detections": detections
    }

def record_detection(detections, source):
    if 'detection_records' not in st.session_state:
        st.session_state.detection_records = []
    for detection in detections:
        st.session_state.detection_records.append({
            "timestamp": detection["timestamp"],
            "source": source,
            "animal": detection["class"],
            "confidence": detection["confidence"]
        })

def display_detection_records():
    if 'detection_records' in st.session_state and st.session_state.detection_records:
        st.subheader("Detection Records")
        df = pd.DataFrame(st.session_state.detection_records)
        st.dataframe(df)
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Detection Records",
            data=csv,
            file_name="animal_detection_records.csv",
            mime="text/csv",
            key=2
        )
    else:
        st.info("No detections recorded yet.")

def export_detection_results():
    st.sidebar.subheader("Export Options")
    if st.sidebar.button("Export All Detections", key=3):
        if 'detection_records' in st.session_state and st.session_state.detection_records:
            df = pd.DataFrame(st.session_state.detection_records)
            csv = df.to_csv(index=False)
            st.sidebar.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"animal_detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key=4
            )
        else:
            st.sidebar.warning("No detections to export.")

def show_detection_statistics():
    if 'detection_records' in st.session_state and st.session_state.detection_records:
        st.subheader("Detection Statistics")
        df = pd.DataFrame(st.session_state.detection_records)
        animal_counts = df["animal"].value_counts()
        st.bar_chart(animal_counts)
        st.subheader("Detections by Source")
        source_counts = df["source"].value_counts()
        st.table(source_counts)

def main():
    st.set_page_config(
        page_title="Deadly Animal Detection Dashboard",
        page_icon="🦁",
        layout="wide"
    )
    st.title("Deadly Animal Detection Dashboard")
    st.markdown("This dashboard detects lions, monkeys, elephants, and leopards in images, videos, and camera feeds.")
    st.sidebar.title("Detection Options")
    detection_mode = st.sidebar.selectbox("Select Detection Mode", ["Camera", "Image Upload", "Video Upload"])
    confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
    export_detection_results()
    if detection_mode == "Camera":
        camera_detection(confidence)
    elif detection_mode == "Image Upload":
        image_detection(confidence)
    elif detection_mode == "Video Upload":
        video_detection(confidence)
    display_detection_records()
    show_detection_statistics()

if __name__ == "__main__":
    main()











