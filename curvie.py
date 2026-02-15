from ultralytics import YOLO
model = YOLO("E:/Data_KHOI/Project_YOLO/runs/segment/Experiment/POLIN/Compare_other/v8/weights/best.pt")
if __name__ == "__main__":
        model.val(
            data="E:/Data_KHOI/Project_YOLO/dataset_cfg/POLIN/polination_test.yaml",
            imgsz= 320,
            batch= 32,
            workers=2,
            device=0,
            seed=0,
            iou=0.75,
        )