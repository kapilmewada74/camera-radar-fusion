from ultralytics import YOLO

model = YOLO(
    r"C:\camera-radar-fusion-project\runs\radiate_camera_yolo11s-3\weights\best.pt"
)

model.val(
    data=r"C:\camera-radar-fusion-project\data\camera_yolo\data.yaml",
    imgsz=640,
    batch=4,
    device=0,
    workers=0,
)