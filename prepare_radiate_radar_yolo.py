import os
import json
import shutil
import random
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(r"C:\camera-radar-fusion-project")

RADIATE_ROOT = PROJECT_ROOT / "data" / "fog_6_0"

RADAR_DIR = RADIATE_ROOT / "Navtech_Cartesian"

ANNOTATION_FILE = RADIATE_ROOT / "annotations" / "annotations.json"

OUTPUT_ROOT = PROJECT_ROOT / "data" / "radar_yolo"

IMAGE_ROOT = OUTPUT_ROOT / "images"
LABEL_ROOT = OUTPUT_ROOT / "labels"


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

CLASS_TO_ID = {
    name: i for i, name in enumerate(CLASS_NAMES)
}


# Dataset split
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42


# ============================================================
# FUNCTIONS
# ============================================================

def create_directories():

    for split in ["train", "val", "test"]:

        (IMAGE_ROOT / split).mkdir(
            parents=True,
            exist_ok=True
        )

        (LABEL_ROOT / split).mkdir(
            parents=True,
            exist_ok=True
        )


def load_annotations():

    if not ANNOTATION_FILE.exists():

        raise FileNotFoundError(
            f"Annotation file not found:\n{ANNOTATION_FILE}"
        )

    with open(
        ANNOTATION_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        annotations = json.load(f)

    return annotations


def get_radar_files():

    files = sorted(
        RADAR_DIR.glob("*.png")
    )

    return files


def annotation_to_yolo(annotation, image_width, image_height):

    bbox = annotation["bbox"]

    position = bbox["position"]

    x = float(position[0])
    y = float(position[1])
    w = float(position[2])
    h = float(position[3])

    # RADIATE annotation is:
    # x, y, width, height
    #
    # x,y = top-left

    x_center = x + w / 2.0
    y_center = y + h / 2.0

    # Normalize
    x_center /= image_width
    y_center /= image_height
    w /= image_width
    h /= image_height

    # Clamp values
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))

    class_name = annotation["class_name"]

    if class_name not in CLASS_TO_ID:
        return None

    class_id = CLASS_TO_ID[class_name]

    return (
        f"{class_id} "
        f"{x_center:.6f} "
        f"{y_center:.6f} "
        f"{w:.6f} "
        f"{h:.6f}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("RADIATE -> YOLO RADAR DATASET PREPARATION")
    print("=" * 70)

    print("\nRADIATE:")
    print(RADIATE_ROOT)

    print("\nRadar:")
    print(RADAR_DIR)

    print("\nAnnotations:")
    print(ANNOTATION_FILE)

    create_directories()

    annotations = load_annotations()

    radar_files = get_radar_files()

    if len(radar_files) == 0:

        raise RuntimeError(
            "No Navtech_Cartesian PNG files found."
        )

    print(f"\nRadar frames found: {len(radar_files)}")
    print(f"Annotation objects: {len(annotations)}")

    # --------------------------------------------------------
    # Create object lookup
    # --------------------------------------------------------

    annotation_lookup = {}

    for obj in annotations:

        object_id = obj["id"]

        annotation_lookup[object_id] = obj

    # --------------------------------------------------------
    # Shuffle frames
    # --------------------------------------------------------

    random.seed(RANDOM_SEED)

    random.shuffle(radar_files)

    total = len(radar_files)

    train_end = int(total * TRAIN_RATIO)

    val_end = train_end + int(total * VAL_RATIO)

    split_files = {

        "train": radar_files[:train_end],

        "val": radar_files[
            train_end:val_end
        ],

        "test": radar_files[
            val_end:
        ],
    }

    print("\nDataset split:")

    for split, files in split_files.items():

        print(
            f"{split}: {len(files)}"
        )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    total_images = 0
    total_objects = 0
    skipped = 0

    class_counts = {
        name: 0
        for name in CLASS_NAMES
    }

    # --------------------------------------------------------
    # Process images
    # --------------------------------------------------------

    for split, files in split_files.items():

        print(
            f"\nProcessing {split}..."
        )

        for index, radar_path in enumerate(files):

            frame_name = radar_path.stem

            # Example:
            # 000001.png
            #
            # RADIATE SDK uses annotation index
            # corresponding to radar frame - 1.

            try:
                frame_id = int(frame_name)
            except ValueError:

                print(
                    f"Skipping invalid frame: {frame_name}"
                )

                skipped += 1
                continue

            annotation_index = frame_id - 1

            # ------------------------------------------------
            # Read radar
            # ------------------------------------------------

            radar = cv2.imread(
                str(radar_path),
                cv2.IMREAD_GRAYSCALE
            )

            if radar is None:

                print(
                    f"Could not read: {radar_path}"
                )

                skipped += 1
                continue

            height, width = radar.shape

            # ------------------------------------------------
            # Get annotations
            # ------------------------------------------------

            frame_annotations = []

            for obj in annotations:

                bboxes = obj.get("bboxes", [])

                if annotation_index < len(bboxes):

                    bbox = bboxes[annotation_index]

                    if bbox:

                        frame_annotations.append({

                            "id": obj["id"],

                            "class_name":
                                obj["class_name"],

                            "bbox": bbox
                        })

            # ------------------------------------------------
            # Skip frames with no annotations
            # ------------------------------------------------

            if len(frame_annotations) == 0:

                skipped += 1
                continue

            # ------------------------------------------------
            # Save radar image
            # ------------------------------------------------

            output_image = (
                IMAGE_ROOT /
                split /
                radar_path.name
            )

            # Keep radar information unchanged.
            # Convert grayscale → 3 channel because
            # YOLO expects normal image input.

            radar_bgr = cv2.cvtColor(
                radar,
                cv2.COLOR_GRAY2BGR
            )

            cv2.imwrite(
                str(output_image),
                radar_bgr
            )

            # ------------------------------------------------
            # YOLO labels
            # ------------------------------------------------

            label_lines = []

            for annotation in frame_annotations:

                yolo_line = annotation_to_yolo(
                    annotation,
                    width,
                    height
                )

                if yolo_line is not None:

                    label_lines.append(
                        yolo_line
                    )

                    class_name = annotation[
                        "class_name"
                    ]

                    class_counts[
                        class_name
                    ] += 1

                    total_objects += 1

            if len(label_lines) == 0:

                skipped += 1
                continue

            output_label = (
                LABEL_ROOT /
                split /
                f"{radar_path.stem}.txt"
            )

            with open(
                output_label,
                "w",
                encoding="utf-8"
            ) as f:

                f.write(
                    "\n".join(label_lines)
                )

            total_images += 1

            if (index + 1) % 50 == 0:

                print(
                    f"  {index + 1}/{len(files)}"
                )

    # --------------------------------------------------------
    # data.yaml
    # --------------------------------------------------------

    yaml_text = f"""path: {OUTPUT_ROOT.as_posix()}
train: images/train
val: images/val
test: images/test

nc: {len(CLASS_NAMES)}

names:
"""

    for index, name in enumerate(CLASS_NAMES):

        yaml_text += (
            f"  {index}: {name}\n"
        )

    with open(
        OUTPUT_ROOT / "data.yaml",
        "w",
        encoding="utf-8"
    ) as f:

        f.write(yaml_text)

    # --------------------------------------------------------
    # Final statistics
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("RADAR DATASET PREPARATION COMPLETE")
    print("=" * 70)

    print(
        f"\nGenerated images : {total_images}"
    )

    print(
        f"Generated objects: {total_objects}"
    )

    print(
        f"Skipped           : {skipped}"
    )

    print("\nClass distribution:")

    for name, count in class_counts.items():

        print(
            f"{name:25s}: {count}"
        )

    print("\nDataset:")
    print(OUTPUT_ROOT)

    print("\ndata.yaml:")
    print(
        OUTPUT_ROOT / "data.yaml"
    )


if __name__ == "__main__":
    main()