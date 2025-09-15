# Cup Detection Model Testing Guide

This guide explains how to test the cup detection model using both static images and live camera feed.

## Prerequisites
- Python 3.6+ installed
- Required packages: `pip install onnxruntime opencv-python numpy loguru`
- Trained ONNX model: `assets/cup_yolox_s.onnx`
- Test images: `assets/cup/` directory

## Testing with Static Images

1. Run the image inference script:
```bash
python tools/onnx_image_demo.py \
  -m assets/cup_yolox_s.onnx \
  -i assets/cup \
  -o detection_results \
  --classes cup cup_body \
  --input_shape 640,640 \
  --score_thr 0.3
```

2. View results:
- Annotated images will be saved in `detection_results/` directory
- Each image will have bounding boxes and confidence scores for detected cups

## Testing with Live Camera

1. Run the webcam inference script:
```bash
python tools/onnx_webcam_demo.py \
  -m assets/cup_yolox_s.onnx \
  --classes cup cup_body \
  --input_shape 640,640 \
  --score_thr 0.3
```

2. Use the application:
- The script will open a window showing live camera feed
- Detected cups will have bounding boxes and labels
- Press 'q' to quit the application

## Command Options
- `-m`: Path to ONNX model file
- `-i`: Input directory for image testing
- `-o`: Output directory for image results
- `--classes`: Custom class names (space separated)
- `--input_shape`: Model input size (width,height)
- `--score_thr`: Confidence threshold (0.0-1.0)
- `--camera_id`: Camera device ID (default: 0)

## Troubleshooting
- If you get "Camera not found" error, try different `--camera_id` values
- For slow performance, reduce input size (e.g., `--input_shape 416,416`)
- For better accuracy, adjust `--score_thr` (higher values = fewer detections)


Run webcam:
python tools/onnx_webcam_demo.py -m assets/cup_yolox_s.onnx --classes cup cup_body

Run images:
python tools/onnx_image_demo.py -m assets/cup_yolox_s.onnx -i assets/cup -o detection_results --classes cup cup_body