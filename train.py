from ultralytics import YOLO
#from ultralytics import RTDETR
# Khởi tạo model từ config (hoặc checkpoint nếu muốn fine-tune)
model = YOLO("E:/Data_KHOI/Project_YOLO/cfg/compare_orther/v8_sw/yolov8-solaf_sw_detect.yaml")
#model = RTDETR("rtdetr-l.yaml")
if __name__ == "__main__":
         model.train(
            data="E:/Data_KHOI/Project_YOLO/dataset_cfg/GLWH/GlobalWheat2020.yaml",
            epochs=400,
            imgsz=320,
            batch=32,
            workers=2,
            device=0,
            optimizer="SGD", # AdamW
            momentum=0.9,
            lr0=0.01,
            weight_decay=0.0005,
            patience=0,  # đảm bảo chạy đủ epoch, không dừng sớm
            seed=0, #resume=True
            project="E:/Data_KHOI/Project_YOLO/runs/segment/Experiment/GlobalWheat2020/orig",
            name=f"big_shift2"  # lưu kết quả theo seed
        )