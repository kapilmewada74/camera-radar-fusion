from ultralytics import YOLO


def main():
    model = YOLO("yolo11s.pt")

    model.train(
        data=r"C:\camera-radar-fusion-project\data\camera_yolo\data.yaml",

        # Real training
        epochs=100,
        imgsz=640,
        batch=4,

        # RTX 3050
        device=0,

        # Important for Windows + Python 3.13
        workers=0,

        # Output
        project=r"C:\camera-radar-fusion-project\runs",
        name="radiate_camera_yolo11s",

        pretrained=True,
        patience=20,

        # Keep validation/checkpoints
        val=True,
        save=True,
        plots=True,
    )


if __name__ == "__main__":
    main()