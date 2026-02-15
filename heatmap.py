import cv2
import torch
import numpy as np
from ultralytics import YOLO
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

class YOLOBoxScoreTarget:
    def __init__(self, box_index=0):
        self.box_index = box_index

    def __call__(self, model_output):
        # model_output: tuple
        preds = model_output[0]  # [num_boxes, 6]
        return preds[self.box_index, 4]  # confidence score

# Load model
yolo = YOLO("yolov8n.pt")
model = yolo.model
model.eval()

# Target layer (backbone cuối)
target_layers = [model.model[-2]]

# Load image
img_path = "E:/Data_KHOI/Project_YOLO/datasets/Test_flower/HOA DUC/Male (5).jpg"
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_norm = img_rgb.astype(np.float32) / 255.0
img_resized = cv2.resize(img_norm, (640, 640))

input_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0)

# YOLO forward
outputs = model(input_tensor)

# Custom target
targets = [YOLOBoxScoreTarget(box_index=0)]

# Grad-CAM
cam = GradCAM(
    model=model,
    target_layers=target_layers,
    use_cuda=False
)

grayscale_cam = cam(
    input_tensor=input_tensor,
    targets=targets
)[0]

# Overlay
visualization = show_cam_on_image(img_resized, grayscale_cam, use_rgb=True)

cv2.imshow("YOLOv8 Grad-CAM", visualization)
cv2.waitKey(0)
