import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from model import UNet
from ultralytics import YOLO
import tempfile
import os
import torchvision.transforms as transforms

# --- SET PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Lane + Object Detection Dashboard",
    page_icon="🛣️",
    layout="centered",
)

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    lane_model = UNet(in_channels=3, out_channels=1)
    checkpoint = torch.load("model/best_unet_model.pth", map_location=torch.device("cpu"))
    lane_model.load_state_dict(checkpoint)
    lane_model.eval()

    yolo_model = YOLO("yolov8n.pt")
    return lane_model, yolo_model

lane_model, yolo_model = load_models()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# --- IMAGE PROCESSING FUNCTION ---
def apply_lane_and_yolo(frame):
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_tensor = transform(pil_image).unsqueeze(0)

    with torch.no_grad():
        output = lane_model(input_tensor)

    mask = output.squeeze().cpu().numpy()
    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
    mask = (mask > 0.5).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    lane_overlay = np.zeros_like(frame)
    lane_overlay[:, :, 1] = mask
    result = cv2.addWeighted(frame, 1, lane_overlay, 0.4, 0)

    detections = yolo_model(frame)
    for det in detections:
        for box in det.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            if conf > 0.5:
                label = f"{yolo_model.names[cls]} {conf:.2f}"
                color = (0, 255, 0)
                cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
                cv2.putText(result, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return result

# --- STREAMLIT UI ---
st.title("🛣️ Lane + Object Detection Dashboard")

st.markdown("""
Detect **lanes and vehicles** in uploaded **images** or **videos** using UNet for lane segmentation and YOLOv8 for object detection.

Developed by **Nishit Popat**, using PyTorch + OpenCV + Ultralytics YOLO.
""")

st.markdown("---")

option = st.radio("Choose input type:", ["Image", "Video"], horizontal=True)

if option == "Image":
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert("RGB")
        frame = np.array(image)

        with st.spinner("Processing image..."):
            result = apply_lane_and_yolo(frame)

        col1, col2 = st.columns(2)
        with col1:
            st.image(frame, caption="Original Image", use_container_width=True)
        with col2:
            st.image(result, caption="Lane + YOLO Detection", use_container_width=True)

elif option == "Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_video is not None:
        temp_input = tempfile.NamedTemporaryFile(delete=False)
        temp_input.write(uploaded_video.read())
        temp_input.close()

        cap = cv2.VideoCapture(temp_input.name)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        temp_output_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))

        with st.spinner("Processing video... this may take a while!"):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                result = apply_lane_and_yolo(frame)
                out.write(result)

        cap.release()
        out.release()

        st.video(temp_output_path)
        with open(temp_output_path, "rb") as f:
            st.download_button("📥 Download Processed Video", f.read(), file_name="output_lane_detected.mp4")

st.markdown("---")
st.markdown("Supports lane detection using UNet and object detection using YOLOv8.")