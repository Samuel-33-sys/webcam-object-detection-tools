# Webcam Object Detection Tools

A collection of real-time object detection scripts using YOLO and your PC's webcam. Adapted from Raspberry Pi/Picamera2 projects to work with standard USB or built-in webcams.

## Features

- **Object Counter** – Counts occurrences of a target object and triggers an action when a threshold is reached.
- **Location Tracker** – Tracks the relative (0–1) position of a target object in the frame.
- **Photo Capture** – Simple tool to capture reference photos with spacebar.

## Requirements

- Python 3.8+
- OpenCV
- Ultralytics YOLO
- A webcam (built-in or USB)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/webcam-object-detection-tools.git
   cd webcam-object-detection-tools
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download YOLO model files:
   - `yoloe-11s-seg.onnx` (small model)
   - `yoloe-11l-seg.onnx` (large model)

   Place them in the same directory as the scripts.

## Usage

### Object Counter
```bash
python object_counter.py
```
Edit the variables at the top of the script to change target object, count threshold, and confidence.

Press `q` to quit.

When the target count is reached, a message prints to console. Add your own action (e.g., send notification, control GPIO) in the marked section.

### Location Tracker
```bash
python location_tracker.py
```
Tracks the center point of detected objects and shows relative coordinates (0.0 to 1.0).

Example action: detect when object enters a specific zone (e.g., bottom-right corner).

Press `q` to quit.

### Photo Capture
```bash
python photo_capture.py
```
Displays live webcam feed.

Press `SPACE` to save the current frame as `image.jpg`.

Press `ESC` to exit.

## Configuration

All scripts include configurable parameters at the top:

| Parameter | Description |
|---|---|
| `TARGET_OBJECT` | Object class name (e.g., `"hand"`, `"person"`, `"bottle"`) |
| `CONFIDENCE_THRESHOLD` | Minimum confidence (0.0–1.0) |
| `TARGET_COUNT` | *(counter only)* Number of detections to trigger action |

## Notes

- Scripts automatically attempt to set 800×800 resolution. If your webcam doesn't support it, the default resolution will be used.
- Color handling is preserved from the original Picamera2 pipeline (RGB ↔ BGR conversions are applied correctly).
- Model files must be in the same folder as the scripts.

## License

MIT
