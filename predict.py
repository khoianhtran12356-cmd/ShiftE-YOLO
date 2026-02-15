from ultralytics import YOLO
import glob
import os

# ---- 1. Load mô hình YOLO (thay bằng model của bạn) ----
model = YOLO("E:/Data_KHOI/Project_YOLO/runs/segment/Experiment/POLIN/Compare_other/sola_sw/weights/best.pt")   # ví dụ: 'yolov8n.pt', 'best.pt'

# ---- 2. Đường dẫn thư mục chứa ảnh ----
#image_folder = "E:/Data_KHOI/Project_YOLO/runs/segment/Experiment/POLIN/test_predict"   # thay đường dẫn thư mục của bạn
image_folder = "E:/Data_KHOI/Project_YOLO/datasets/hoa_test1"   # thay đường dẫn thư mục của bạn

# ---- 3. Lấy danh sách ảnh ----
image_paths = glob.glob(os.path.join(image_folder, "*.jpg"))  # hoặc *.png

# ---- 4. Chỉ predict 4 ảnh đầu tiên ----
image_paths = image_paths[:]

print("Danh sách ảnh sẽ predict:")
for img in image_paths:
    print(" -", img)

# ---- 5. Predict ----
if __name__ == "__main__":
    results = model.predict(
    source=image_paths,
    imgsz=320,
    device=0,
    save=True,       # lưu ảnh có bounding box
    conf=0.5,        # confidence threshold
    iou=0.7)


print("/nDone! Kết quả nằm trong thư mục 'runs/detect/'.")
