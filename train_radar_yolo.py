from ultralytics import YOLO


def main():

    model = YOLO("yolo11s.pt")

    model.train(

        data=r"C:\camera-radar-fusion-project\data\radar_yolo\data.yaml",

        epochs=100,

        imgsz=640,

        batch=4,

        device=0,

        # Important for Windows + Python 3.13
        workers=0,

        project=r"C:\camera-radar-fusion-project\runs",

        name="radiate_radar_yolo11s",

        pretrained=True,

        patience=20,

        val=True,

        save=True,

        plots=True,
    )


if __name__ == "__main__":
    main()