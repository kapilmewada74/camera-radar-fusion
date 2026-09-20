from pathlib import Path
import cv2
import random

ROOT = Path(r"C:\camera-radar-fusion-project\data\camera_yolo")

CLASS_NAMES = [
    "car",
    "van",
    "truck",
    "bus",
    "motorbike",
    "bicycle",
    "pedestrian",
    "group_of_pedestrians",
]

split = "train"

image_dir = ROOT / "images" / split
label_dir = ROOT / "labels" / split

images = list(image_dir.glob("*"))

if not images:
    raise RuntimeError("No images found.")

random.shuffle(images)

for image_path in images[:10]:

    label_path = label_dir / f"{image_path.stem}.txt"

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not read: {image_path}")
        continue

    h, w = image.shape[:2]

    if label_path.exists():

        with open(label_path, "r") as f:
            lines = f.readlines()

        for line in lines:

            parts = line.strip().split()

            if len(parts) != 5:
                continue

            class_id = int(parts[0])
            xc, yc, bw, bh = map(float, parts[1:])

            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)

            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)

            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))

            name = (
                CLASS_NAMES[class_id]
                if 0 <= class_id < len(CLASS_NAMES)
                else str(class_id)
            )

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            cv2.putText(
                image,
                name,
                (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

    cv2.imshow("RADIATE YOLO Annotation Check", image)

    print(f"Showing: {image_path.name}")
    print("Press any key for next image.")
    print("Press ESC to exit.")

    key = cv2.waitKey(0)

    if key == 27:
        break

cv2.destroyAllWindows()