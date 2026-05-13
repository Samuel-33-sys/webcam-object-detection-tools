import cv2
from ultralytics import YOLO

TARGET_OBJECT = "hand"
CONFIDENCE_THRESHOLD = 0.2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)

model = YOLO("yoloe-11l-seg.onnx")
class_names = model.names

print(f"Tracking location of: {TARGET_OBJECT}")
print(f"Minimum confidence: {CONFIDENCE_THRESHOLD}")
print("Press 'q' to quit")

while True:
    ret, frame_bgr = cap.read()
    if not ret:
        break

    frame_height, frame_width = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = model.predict(frame_rgb)

    object_locations = []

    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        detected_classes = results[0].boxes.cls.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()

        for i, class_id in enumerate(detected_classes):
            class_name = class_names[int(class_id)]
            confidence = confidences[i]
            if class_name.lower() == TARGET_OBJECT.lower() and confidence >= CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = boxes[i]
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                relative_x = center_x / frame_width
                relative_y = center_y / frame_height
                object_locations.append({
                    'x': relative_x, 'y': relative_y,
                    'pixel_x': int(center_x), 'pixel_y': int(center_y),
                    'confidence': confidence
                })

    for i, loc in enumerate(object_locations):
        print(f"{TARGET_OBJECT} #{i+1} at ({loc['x']:.3f}, {loc['y']:.3f}) - conf: {loc['confidence']:.3f}")
        # Example custom zone trigger:
        if loc['x'] > 0.5 and loc['y'] > 0.5:
            print(f"HIGH CONFIDENCE {TARGET_OBJECT} in bottom right corner!")

    annotated_frame = results[0].plot(boxes=True, masks=False)

    for loc in object_locations:
        px, py = loc['pixel_x'], loc['pixel_y']
        cv2.circle(annotated_frame, (px, py), 5, (0, 255, 255), -1)
        coord_text = f"({loc['x']:.2f}, {loc['y']:.2f}) {loc['confidence']:.2f}"
        cv2.putText(annotated_frame, coord_text, (px + 10, py - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    inference_time = results[0].speed['inference']
    fps = 1000 / inference_time
    status_text = f"Tracking: {TARGET_OBJECT} | Found: {len(object_locations)} | FPS: {fps:.1f} | Min Conf: {CONFIDENCE_THRESHOLD}"
    cv2.putText(annotated_frame, status_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Object Location Tracker", annotated_frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
