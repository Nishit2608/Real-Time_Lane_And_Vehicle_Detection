import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from model import UNet
from ultralytics import YOLO

def get_traffic_light_color(frame, x1, y1, x2, y2):
    light = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(light, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([90, 255, 255])

    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    red_pixels = cv2.countNonZero(mask_red)
    yellow_pixels = cv2.countNonZero(mask_yellow)
    green_pixels = cv2.countNonZero(mask_green)

    max_pixels = max(red_pixels, yellow_pixels, green_pixels)
    if max_pixels == red_pixels:
        return "Red"
    elif max_pixels == yellow_pixels:
        return "Yellow"
    elif max_pixels == green_pixels:
        return "Green"
    else:
        return "Unknown"

def process_frame(frame, lane_model, yolo_model, transform):
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_tensor = transform(pil_image).unsqueeze(0)

    with torch.no_grad():
        output = lane_model(input_tensor)

    mask = output.squeeze().cpu().numpy()
    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
    mask = (mask > 0.5).astype(np.uint8) * 255
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

    lane_colored = np.zeros_like(frame)
    lane_colored[:, :, 1] = mask

    result = cv2.addWeighted(frame, 1, lane_colored, 0.4, 0)

    yolo_results = yolo_model(frame)
    for r in yolo_results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            if conf > 0.5:
                class_name = yolo_model.names[cls]
                color = (0, 255, 0)
                label = f'{class_name} {conf:.2f}'

                if class_name == "traffic light":
                    light_color = get_traffic_light_color(frame, x1, y1, x2, y2)
                    label = f'Traffic Light ({light_color}) {conf:.2f}'
                    color = {
                        "Red": (0, 0, 255),
                        "Yellow": (0, 255, 255),
                        "Green": (0, 255, 0)
                    }.get(light_color, (255, 255, 255))

                cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
                cv2.putText(result, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return result

def main():
    lane_model = UNet(in_channels=3, out_channels=1)
    checkpoint = torch.load(r"model\best_unet_model.pth", map_location=torch.device("cpu"))
    lane_model.load_state_dict(checkpoint)
    lane_model.eval()

    yolo_model = YOLO("yolov8n.pt")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    cap = cv2.VideoCapture("nD_1.mp4")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter("output_yolo.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result_frame = process_frame(frame, lane_model, yolo_model, transform)
        out.write(result_frame)

        cv2.imshow("Lane Detection and Object Detection", result_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
