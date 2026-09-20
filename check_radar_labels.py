import random
from pathlib import Path

import cv2


ROOT = Path(
    r"C:\camera-radar-fusion-project\data\radar_yolo"
)

IMAGE_DIR = ROOT / "images" / "train"
LABEL_DIR = ROOT / "labels" / "train"


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


def main():

    images = list(
        IMAGE_DIR.glob("*.png")
    )

    random.seed(42)

    selected = random.sample(
        images,
        min(10, len(images))
    )

    output_dir = Path(
        r"C:\camera-radar-fusion-project\runs\radar_label_check"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for image_path in selected:

        image = cv2.imread(
            str(image_path)
        )

        h, w = image.shape[:2]

        label_path = (
            LABEL_DIR /
            f"{image_path.stem}.txt"
        )

        if not label_path.exists():
            continue

        with open(
            label_path,
            "r"
        ) as f:

            lines = f.readlines()

        for line in lines:

            parts = line.strip().split()

            if len(parts) != 5:
                continue

            class_id = int(parts[0])

            xc = float(parts[1]) * w
            yc = float(parts[2]) * h
            bw = float(parts[3]) * w
            bh = float(parts[4]) * h

            x1 = int(xc - bw / 2)
            y1 = int(yc - bh / 2)
            x2 = int(xc + bw / 2)
            y2 = int(yc + bh / 2)

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (255, 255, 255),
                2
            )

            cv2.putText(
                image,
                CLASS_NAMES[class_id],
                (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        output_path = (
            output_dir /
            image_path.name
        )

        cv2.imwrite(
            str(output_path),
            image
        )

    print(
        f"Saved visualizations to:\n{output_dir}"
    )


if __name__ == "__main__":
    main()