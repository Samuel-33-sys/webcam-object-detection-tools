import cv2
from ultralytics import YOLO

TARGET_OBJECT = "hand"        # What object to look for (e.g., "person", "bottle", "cup")
TARGET_COUNT = 1              # How many of that object should trigger the action
CONFIDENCE_THRESHOLD = 0.2    # Minimum confidence score (0.0 to 1.0)

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)

model = YOLO("yoloe-11s-seg.onnx")
class_names = model.names

print(f"Looking for {TARGET_COUNT} {TARGET_OBJECT}(s)")
print(f"Minimum confidence: {CONFIDENCE_THRESHOLD}")
print("Press 'q' to quit")

while True:
    ret, frame_bgr = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = model.predict(frame_rgb)

    object_count = 0
    confident_objects = []

    if results[0].boxes is not None:
        detected_classes = results[0].boxes.cls.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()

        for i, class_id in enumerate(detected_classes):
            class_name = class_names[int(class_id)]
            confidence = confidences[i]
            if class_name.lower() == TARGET_OBJECT.lower() and confidence >= CONFIDENCE_THRESHOLD:
                object_count += 1
                confident_objects.append({'class_name': class_name, 'confidence': confidence})

    if object_count >= TARGET_COUNT:
        print(f"Target number of objects detected! ({object_count} confident {TARGET_OBJECT}(s))")
        for i, obj in enumerate(confident_objects):
            print(f"  {TARGET_OBJECT} #{i+1}: {obj['confidence']:.3f} confidence")
        # ADD YOUR CUSTOM ACTION HERE

    annotated_frame = results[0].plot(boxes=True, masks=False)
    inference_time = results[0].speed['inference']
    fps = 1000 / inference_time

    status_text = f"Looking for: {TARGET_COUNT} {TARGET_OBJECT}(s) | Found: {object_count} | FPS: {fps:.1f} | Min Conf: {CONFIDENCE_THRESHOLD}"
    cv2.putText(annotated_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    if object_count >= TARGET_COUNT:
        cv2.putText(annotated_frame, "TARGET REACHED!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)
        for i, obj in enumerate(confident_objects):
            conf_text = f"{TARGET_OBJECT} #{i+1}: {obj['confidence']:.2f}"
            cv2.putText(annotated_frame, conf_text, (10, 110 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Object Counter", annotated_frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
