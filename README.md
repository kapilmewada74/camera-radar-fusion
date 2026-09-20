# FogFusion: SDK-free camera-only baseline

This project trains a camera-only object detector on raw RADIATE left-camera images without the RADIATE SDK.

## Prepare the images

```powershell
python prepare_radiate_camera_yolo.py
```

It creates `radiate_camera_yolo/` with these splits:

- First 70% of `fog_6_0`: train
- Next 15% of `fog_6_0`: validation
- Final 15% of `fog_6_0`: excluded buffer
- All `fog_8_0`: final test

## Label the images

Use CVAT, LabelImg, or Roboflow. Export YOLO detection labels and put a matching `.txt` label file for every image into the equivalent `labels/train`, `labels/val`, or `labels/test` folder. Empty `.txt` files are valid for images containing no objects.

## Train and test

```powershell
python -m pip install -r requirements.txt
python train_camera.py
python validate_camera.py --split val
python validate_camera.py --split test
```

The model is saved at `outputs/camera_only/weights/best.pt`.
