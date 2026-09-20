"""
RADIATE -> YOLO CAMERA DATASET PREPARATION

Input:
    C:/camera-radar-fusion-project/data/fog_6_0

Uses:
    - RADIATE ZED stereo camera
    - RADIATE annotations
    - RADIATE calibration
    - RADIATE timestamps

Output:
    C:/camera-radar-fusion-project/data/camera_yolo/

Dataset:
    images/train
    images/val
    images/test

    labels/train
    labels/val
    labels/test

Important:
    RADIATE raw zed_left images are unrectified.
    Therefore this script uses the RADIATE SDK to:
        1. synchronize camera/radar/lidar
        2. rectify the left camera
        3. project RADIATE annotations into the rectified camera
        4. convert the resulting boxes to YOLO format
"""

import os
import sys
import shutil
import tempfile
from pathlib import Path

import cv2
import yaml
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(r"C:\camera-radar-fusion-project")

RADIATE_SEQUENCE = (
    PROJECT_ROOT
    / "data"
    / "fog_6_0"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "camera_yolo"
)

SDK_ROOT = (
    PROJECT_ROOT
    / "radiate_sdk"
)


# ============================================================
# SETTINGS
# ============================================================

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Minimum width/height of a valid box
MIN_BOX_WIDTH = 2
MIN_BOX_HEIGHT = 2

# Keep only objects whose projected box has enough visible area
MIN_VISIBLE_AREA_RATIO = 0.20

# Classes supported by RADIATE
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
    name: idx
    for idx, name in enumerate(CLASS_NAMES)
}


# ============================================================
# PRINT HELPERS
# ============================================================

def print_line():
    print("=" * 75)


def info(message):
    print(f"[INFO] {message}")


def warning(message):
    print(f"[WARNING] {message}")


def error(message):
    print(f"[ERROR] {message}")


# ============================================================
# CHECK PATHS
# ============================================================

def check_paths():

    print_line()
    print("CHECKING RADIATE DATASET")
    print_line()

    if not PROJECT_ROOT.exists():
        raise FileNotFoundError(
            f"Project root does not exist:\n{PROJECT_ROOT}"
        )

    if not RADIATE_SEQUENCE.exists():
        raise FileNotFoundError(
            f"RADIATE sequence does not exist:\n{RADIATE_SEQUENCE}"
        )

    required_paths = [
        RADIATE_SEQUENCE / "annotations",
        RADIATE_SEQUENCE / "zed_left",
        RADIATE_SEQUENCE / "zed_right",
        RADIATE_SEQUENCE / "Navtech_Cartesian",
        RADIATE_SEQUENCE / "velo_lidar",
        RADIATE_SEQUENCE / "zed_left.txt",
        RADIATE_SEQUENCE / "Navtech_Cartesian.txt",
        RADIATE_SEQUENCE / "velo_lidar.txt",
    ]

    for path in required_paths:

        if not path.exists():

            raise FileNotFoundError(
                f"Required RADIATE file/folder missing:\n{path}"
            )

        print(f"[OK] {path}")

    annotation_file = (
        RADIATE_SEQUENCE
        / "annotations"
        / "annotations.json"
    )

    if not annotation_file.exists():

        raise FileNotFoundError(
            f"Annotation file missing:\n{annotation_file}"
        )

    print(f"[OK] {annotation_file}")


# ============================================================
# IMPORT RADIATE SDK
# ============================================================

def import_radiate_sdk():

    print_line()
    print("LOADING RADIATE SDK")
    print_line()

    if not SDK_ROOT.exists():

        raise FileNotFoundError(
            "\nRADIATE SDK was not found.\n\n"
            "Run:\n\n"
            "git clone https://github.com/marcelsheeny/radiate_sdk.git\n\n"
            "inside:\n"
            f"{PROJECT_ROOT}\n"
        )

    # Add SDK root to Python path
    if str(SDK_ROOT) not in sys.path:
        sys.path.insert(0, str(SDK_ROOT))

    # radiate.py imports utils.calibration,
    # therefore SDK root must be on sys.path.
    try:

        import radiate

        print("[OK] RADIATE SDK imported")

        return radiate

    except Exception as exc:

        raise RuntimeError(
            "\nCould not import the RADIATE SDK.\n\n"
            "Try:\n"
            "pip install -r "
            f"{SDK_ROOT}\\requirements.txt\n\n"
            f"Original error:\n{exc}"
        )


# ============================================================
# LOAD SDK CONFIG
# ============================================================

def prepare_sdk_config():

    """
    Load the official SDK config and override only the settings
    required for camera dataset generation.
    """

    original_config = SDK_ROOT / "config" / "config.yaml"

    if not original_config.exists():

        raise FileNotFoundError(
            f"RADIATE SDK config not found:\n{original_config}"
        )

    with open(
        original_config,
        "r",
        encoding="utf-8"
    ) as f:

        config = yaml.safe_load(f)

    if config is None:
        config = {}

    # --------------------------------------------------------
    # Camera
    # --------------------------------------------------------

    config["use_camera_left_raw"] = False
    config["use_camera_right_raw"] = False

    config["use_camera_left_rect"] = True
    config["use_camera_right_rect"] = False

    # --------------------------------------------------------
    # Radar
    # --------------------------------------------------------

    config["use_radar_cartesian"] = False
    config["use_radar_polar"] = False

    # --------------------------------------------------------
    # LiDAR
    # --------------------------------------------------------

    # IMPORTANT:
    # The SDK uses the LiDAR BEV path internally to obtain
    # the annotation coordinate transformation before
    # projecting annotations into the camera.
    config["use_lidar_bev_image"] = True

    config["use_proj_lidar_left"] = False
    config["use_proj_lidar_right"] = False

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    config["save_images"] = False

    # --------------------------------------------------------
    # Annotation interpolation
    # --------------------------------------------------------

    config["interpolate_bboxes"] = False

    # --------------------------------------------------------
    # Output folder
    # --------------------------------------------------------

    temp_output = (
        PROJECT_ROOT
        / "_radiate_sdk_temp_output"
    )

    temp_output.mkdir(
        parents=True,
        exist_ok=True
    )

    config["calib_file"] = str(
    SDK_ROOT / "config" / "default-calib.yaml"
    )

    # --------------------------------------------------------
    # Write temporary config
    # --------------------------------------------------------

    temp_config = (
        PROJECT_ROOT
        / "_radiate_camera_config.yaml"
    )

    with open(
        temp_config,
        "w",
        encoding="utf-8"
    ) as f:

        yaml.safe_dump(
            config,
            f,
            sort_keys=False
        )

    return temp_config


# ============================================================
# CREATE YOLO DIRECTORIES
# ============================================================

def create_output_directories():

    print_line()
    print("CREATING YOLO DATASET DIRECTORIES")
    print_line()

    for split in ["train", "val", "test"]:

        image_dir = (
            OUTPUT_ROOT
            / "images"
            / split
        )

        label_dir = (
            OUTPUT_ROOT
            / "labels"
            / split
        )

        image_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        label_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        print(f"[OK] {image_dir}")
        print(f"[OK] {label_dir}")


# ============================================================
# GET CAMERA FRAME IDS
# ============================================================

def load_camera_timestamps(sequence):

    """
    RADIATE SDK timestamp format:

        Frame: 000001 Time: 1574859771.744660272
    """

    timestamp_file = (
        RADIATE_SEQUENCE
        / "zed_left.txt"
    )

    timestamps = []

    with open(
        timestamp_file,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) < 4:
                continue

            try:

                frame_id = int(parts[1])
                timestamp = float(parts[3])

                timestamps.append(
                    {
                        "frame": frame_id,
                        "time": timestamp
                    }
                )

            except (ValueError, IndexError):
                continue

    return timestamps


# ============================================================
# GET RADAR TIMESTAMPS
# ============================================================

def load_radar_timestamps():

    timestamp_file = (
        RADIATE_SEQUENCE
        / "Navtech_Cartesian.txt"
    )

    timestamps = []

    with open(
        timestamp_file,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) < 4:
                continue

            try:

                frame_id = int(parts[1])
                timestamp = float(parts[3])

                timestamps.append(
                    {
                        "frame": frame_id,
                        "time": timestamp
                    }
                )

            except (ValueError, IndexError):
                continue

    return timestamps


# ============================================================
# SPLIT SEQUENTIALLY
# ============================================================

def create_split_indices(num_samples):

    """
    For a single RADIATE sequence, split sequentially.

    NOTE:
    For a final research dataset containing many RADIATE
    sequences, sequence-level splitting is preferable.
    """

    train_end = int(
        num_samples * TRAIN_RATIO
    )

    val_end = train_end + int(
        num_samples * VAL_RATIO
    )

    indices = list(range(num_samples))

    train_indices = indices[:train_end]

    val_indices = indices[
        train_end:val_end
    ]

    test_indices = indices[
        val_end:
    ]

    return (
        set(train_indices),
        set(val_indices),
        set(test_indices)
    )


# ============================================================
# GET SPLIT
# ============================================================

def get_split(index, train_indices, val_indices, test_indices):

    if index in train_indices:
        return "train"

    if index in val_indices:
        return "val"

    if index in test_indices:
        return "test"

    return None


# ============================================================
# CLIP BOX TO IMAGE
# ============================================================

def clip_box(
    x1,
    y1,
    x2,
    y2,
    image_width,
    image_height
):

    x1 = max(
        0,
        min(
            image_width - 1,
            x1
        )
    )

    y1 = max(
        0,
        min(
            image_height - 1,
            y1
        )
    )

    x2 = max(
        0,
        min(
            image_width - 1,
            x2
        )
    )

    y2 = max(
        0,
        min(
            image_height - 1,
            y2
        )
    )

    return x1, y1, x2, y2


# ============================================================
# CONVERT PROJECTED BOX TO YOLO
# ============================================================

def projected_points_to_yolo(
    points,
    image_width,
    image_height
):

    """
    points:
        Nx2 projected camera points

    Returns:
        x_center
        y_center
        width
        height

    normalized to [0,1]
    """

    if points is None:
        return None

    points = np.asarray(points)

    if points.size == 0:
        return None

    if points.ndim != 2:
        return None

    if points.shape[1] != 2:
        return None

    # Remove invalid values
    points = points[
        np.isfinite(points).all(axis=1)
    ]

    if len(points) == 0:
        return None

    x1 = float(
        np.min(points[:, 0])
    )

    y1 = float(
        np.min(points[:, 1])
    )

    x2 = float(
        np.max(points[:, 0])
    )

    y2 = float(
        np.max(points[:, 1])
    )

    # Original dimensions
    original_width = x2 - x1
    original_height = y2 - y1

    if (
        original_width < MIN_BOX_WIDTH
        or
        original_height < MIN_BOX_HEIGHT
    ):
        return None

    # Calculate visible intersection
    clipped_x1, clipped_y1, clipped_x2, clipped_y2 = clip_box(
        x1,
        y1,
        x2,
        y2,
        image_width,
        image_height
    )

    visible_width = (
        clipped_x2 - clipped_x1
    )

    visible_height = (
        clipped_y2 - clipped_y1
    )

    if (
        visible_width <= 0
        or
        visible_height <= 0
    ):
        return None

    original_area = (
        original_width
        *
        original_height
    )

    visible_area = (
        visible_width
        *
        visible_height
    )

    visible_ratio = (
        visible_area
        /
        max(original_area, 1e-6)
    )

    if visible_ratio < MIN_VISIBLE_AREA_RATIO:
        return None

    # YOLO box
    x_center = (
        clipped_x1
        +
        clipped_x2
    ) / 2.0

    y_center = (
        clipped_y1
        +
        clipped_y2
    ) / 2.0

    width = (
        clipped_x2
        -
        clipped_x1
    )

    height = (
        clipped_y2
        -
        clipped_y1
    )

    # Normalize
    x_center /= image_width
    y_center /= image_height
    width /= image_width
    height /= image_height

    # Safety clipping
    x_center = np.clip(
        x_center,
        0.0,
        1.0
    )

    y_center = np.clip(
        y_center,
        0.0,
        1.0
    )

    width = np.clip(
        width,
        0.0,
        1.0
    )

    height = np.clip(
        height,
        0.0,
        1.0
    )

    return (
        x_center,
        y_center,
        width,
        height
    )


# ============================================================
# SAVE YOLO LABEL
# ============================================================

def save_yolo_label(
    label_path,
    labels
):

    with open(
        label_path,
        "w",
        encoding="utf-8"
    ) as f:

        for label in labels:

            class_id = label["class_id"]

            x_center = label["x_center"]
            y_center = label["y_center"]
            width = label["width"]
            height = label["height"]

            f.write(
                f"{class_id} "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{width:.6f} "
                f"{height:.6f}\n"
            )


# ============================================================
# SAVE DATASET YAML
# ============================================================

def save_dataset_yaml():

    yaml_path = (
        OUTPUT_ROOT
        / "data.yaml"
    )

    data = {
        "path": str(
            OUTPUT_ROOT
        ),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }

    with open(
        yaml_path,
        "w",
        encoding="utf-8"
    ) as f:

        yaml.safe_dump(
            data,
            f,
            sort_keys=False
        )

    print(
        f"[OK] Created: {yaml_path}"
    )


# ============================================================
# CREATE VISUAL DEBUG IMAGES
# ============================================================

def create_debug_image(
    image,
    labels
):

    debug = image.copy()

    h, w = debug.shape[:2]

    for item in labels:

        class_id = item["class_id"]

        x_center = item["x_center"] * w
        y_center = item["y_center"] * h

        box_width = item["width"] * w
        box_height = item["height"] * h

        x1 = int(
            x_center - box_width / 2
        )

        y1 = int(
            y_center - box_height / 2
        )

        x2 = int(
            x_center + box_width / 2
        )

        y2 = int(
            y_center + box_height / 2
        )

        cv2.rectangle(
            debug,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            debug,
            CLASS_NAMES[class_id],
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )

    return debug


# ============================================================
# MAIN DATASET CREATION
# ============================================================

def create_dataset(radiate):

    print_line()
    print("STARTING RADIATE CAMERA DATASET CREATION")
    print_line()

    info(
        f"Sequence:\n{RADIATE_SEQUENCE}"
    )

    info(
        f"Output:\n{OUTPUT_ROOT}"
    )

    # --------------------------------------------------------
    # Create config
    # --------------------------------------------------------

    temp_config = prepare_sdk_config()

    info(
        f"Temporary SDK config:\n{temp_config}"
    )

    # --------------------------------------------------------
    # Load RADIATE sequence
    # --------------------------------------------------------

    print()
    info("Loading RADIATE Sequence...")

    try:

        sequence = radiate.Sequence(
            str(RADIATE_SEQUENCE),
            config_file=str(temp_config)
        )

    except Exception as exc:

        raise RuntimeError(
            "\nRADIATE Sequence could not be loaded.\n\n"
            "This usually means the SDK calibration/configuration "
            "is not installed correctly.\n\n"
            f"Error:\n{exc}"
        )

    print(
        "[OK] RADIATE sequence loaded"
    )

    # --------------------------------------------------------
    # Get camera timestamps
    # --------------------------------------------------------

    camera_timestamps = (
        load_camera_timestamps(sequence)
    )

    if len(camera_timestamps) == 0:

        raise RuntimeError(
            "No camera timestamps found."
        )

    info(
        f"Camera timestamps: "
        f"{len(camera_timestamps)}"
    )

    # --------------------------------------------------------
    # Get radar timestamps
    # --------------------------------------------------------

    radar_timestamps = (
        load_radar_timestamps()
    )

    if len(radar_timestamps) == 0:

        raise RuntimeError(
            "No radar timestamps found."
        )

    info(
        f"Radar timestamps: "
        f"{len(radar_timestamps)}"
    )

    # --------------------------------------------------------
    # Use RADAR timestamps because RADIATE annotations
    # are indexed according to the radar annotation stream.
    # --------------------------------------------------------

    samples = []

    used_camera_frames = set()

    for radar in radar_timestamps:

        radar_time = radar["time"]

        # Find nearest camera
        camera_times = np.array(
            [
                item["time"]
                for item in camera_timestamps
            ],
            dtype=np.float64
        )

        camera_index = int(
            np.argmin(
                np.abs(
                    camera_times
                    -
                    radar_time
                )
            )
        )

        camera_frame = (
            camera_timestamps[
                camera_index
            ]["frame"]
        )

        camera_time = (
            camera_timestamps[
                camera_index
            ]["time"]
        )

        # Avoid processing the same camera image twice
        if camera_frame in used_camera_frames:
            continue

        used_camera_frames.add(
            camera_frame
        )

        samples.append(
            {
                "radar_frame": radar["frame"],
                "radar_time": radar_time,
                "camera_frame": camera_frame,
                "camera_time": camera_time,
            }
        )

    if len(samples) == 0:

        raise RuntimeError(
            "No synchronized camera/radar samples found."
        )

    info(
        f"Unique synchronized samples: "
        f"{len(samples)}"
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    (
        train_indices,
        val_indices,
        test_indices
    ) = create_split_indices(
        len(samples)
    )

    print()
    print(
        f"Train samples: {len(train_indices)}"
    )

    print(
        f"Val samples:   {len(val_indices)}"
    )

    print(
        f"Test samples:  {len(test_indices)}"
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    statistics = {
        "train": 0,
        "val": 0,
        "test": 0,
        "images_written": 0,
        "labels_written": 0,
        "objects": 0,
        "skipped": 0,
    }

    # --------------------------------------------------------
    # Process samples
    # --------------------------------------------------------

    for sample_index, sample in enumerate(samples):

        split = get_split(
            sample_index,
            train_indices,
            val_indices,
            test_indices
        )

        if split is None:
            statistics["skipped"] += 1
            continue

        camera_time = sample["camera_time"]

        try:

            output = sequence.get_from_timestamp(
                camera_time,
                get_sensors=True,
                get_annotations=True
            )

        except Exception as exc:

            warning(
                f"Failed sample "
                f"{sample_index}: {exc}"
            )

            statistics["skipped"] += 1
            continue

        if not output:

            warning(
                f"No RADIATE output for "
                f"sample {sample_index}"
            )

            statistics["skipped"] += 1
            continue

        # ----------------------------------------------------
        # Camera image
        # ----------------------------------------------------

        sensors = output.get(
            "sensors",
            {}
        )

        annotations = output.get(
            "annotations",
            {}
        )

        if (
            "camera_left_rect"
            not in sensors
        ):

            warning(
                f"No rectified camera image "
                f"for sample {sample_index}"
            )

            statistics["skipped"] += 1
            continue

        if (
            "camera_left_rect"
            not in annotations
        ):

            warning(
                f"No camera annotations "
                f"for sample {sample_index}"
            )

            statistics["skipped"] += 1
            continue

        image = sensors[
            "camera_left_rect"
        ]

        camera_annotations = annotations[
            "camera_left_rect"
        ]

        if image is None:

            statistics["skipped"] += 1
            continue

        image_height, image_width = (
            image.shape[:2]
        )

        # ----------------------------------------------------
        # Convert projected camera boxes to YOLO
        # ----------------------------------------------------

        yolo_labels = []

        for obj in camera_annotations:

            class_name = obj.get(
                "class_name",
                ""
            )

            if class_name not in CLASS_TO_ID:

                warning(
                    f"Unknown class: "
                    f"{class_name}"
                )

                continue

            bbox_points = obj.get(
                "bbox_3d",
                None
            )

            yolo_box = projected_points_to_yolo(
                bbox_points,
                image_width,
                image_height
            )

            if yolo_box is None:
                continue

            (
                x_center,
                y_center,
                width,
                height
            ) = yolo_box

            yolo_labels.append(
                {
                    "class_id":
                        CLASS_TO_ID[class_name],

                    "x_center":
                        x_center,

                    "y_center":
                        y_center,

                    "width":
                        width,

                    "height":
                        height,
                }
            )

        # ----------------------------------------------------
        # Skip images with no usable annotation
        # ----------------------------------------------------

        if len(yolo_labels) == 0:

            statistics["skipped"] += 1
            continue

        # ----------------------------------------------------
        # Filename
        # ----------------------------------------------------

        filename = (
            f"fog_6_0_"
            f"{sample['camera_frame']:06d}"
            ".png"
        )

        image_path = (
            OUTPUT_ROOT
            / "images"
            / split
            / filename
        )

        label_path = (
            OUTPUT_ROOT
            / "labels"
            / split
            / filename.replace(
                ".png",
                ".txt"
            )
        )

        # ----------------------------------------------------
        # Save image
        # ----------------------------------------------------

        success = cv2.imwrite(
            str(image_path),
            image
        )

        if not success:

            warning(
                f"Could not write image: "
                f"{image_path}"
            )

            statistics["skipped"] += 1
            continue

        # ----------------------------------------------------
        # Save YOLO label
        # ----------------------------------------------------

        save_yolo_label(
            label_path,
            yolo_labels
        )

        statistics[
            "images_written"
        ] += 1

        statistics[
            "labels_written"
        ] += 1

        statistics[
            "objects"
        ] += len(yolo_labels)

        statistics[
            split
        ] += 1

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            sample_index % 25 == 0
            or
            sample_index == len(samples) - 1
        ):

            print(
                f"[{sample_index + 1}/"
                f"{len(samples)}] "
                f"{split.upper()} "
                f"| frame="
                f"{sample['camera_frame']:06d} "
                f"| objects="
                f"{len(yolo_labels)}"
            )

    # --------------------------------------------------------
    # Save dataset YAML
    # --------------------------------------------------------

    save_dataset_yaml()

    # --------------------------------------------------------
    # Print statistics
    # --------------------------------------------------------

    print()
    print_line()
    print("DATASET CREATION COMPLETE")
    print_line()

    print(
        f"Train images : "
        f"{statistics['train']}"
    )

    print(
        f"Val images   : "
        f"{statistics['val']}"
    )

    print(
        f"Test images  : "
        f"{statistics['test']}"
    )

    print(
        f"Total images : "
        f"{statistics['images_written']}"
    )

    print(
        f"Total objects: "
        f"{statistics['objects']}"
    )

    print(
        f"Skipped      : "
        f"{statistics['skipped']}"
    )

    print()
    print(
        "Camera YOLO dataset:"
    )

    print(
        OUTPUT_ROOT
    )

    print()
    print(
        "YOLO YAML:"
    )

    print(
        OUTPUT_ROOT / "data.yaml"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print_line()
    print(
        "RADIATE -> YOLO CAMERA DATASET"
    )
    print_line()

    try:

        # -----------------------------------------------
        # Check dataset
        # -----------------------------------------------

        check_paths()

        # -----------------------------------------------
        # Import SDK
        # -----------------------------------------------

        radiate = (
            import_radiate_sdk()
        )

        # -----------------------------------------------
        # Create directories
        # -----------------------------------------------

        create_output_directories()

        # -----------------------------------------------
        # Create dataset
        # -----------------------------------------------

        create_dataset(
            radiate
        )

        print()
        print_line()
        print("SUCCESS")
        print_line()

    except KeyboardInterrupt:

        print()
        print(
            "\nProcess stopped by user."
        )

        sys.exit(1)

    except Exception as exc:

        print()
        print_line()
        error("DATASET CREATION FAILED")
        print_line()

        print()
        print(str(exc))

        print()

        sys.exit(1)


if __name__ == "__main__":
    main()