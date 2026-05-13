import cv2

filename = "image.jpg"

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)

print("Reference Photo Capture Tool")
print("Position your object and press SPACE to capture")
print("Press ESC to exit")

while True:
    ret, frame_bgr = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)  # Match original behavior
    cv2.putText(frame_rgb, "SPACE: Take Photo | ESC: Exit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Capture Reference Photo", frame_rgb)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        cv2.imwrite(filename, frame_rgb)
        print(f"Photo saved: {filename}")
        confirm_frame = frame_rgb.copy()
        cv2.putText(confirm_frame, f"Saved: {filename}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Capture Reference Photo", confirm_frame)
        cv2.waitKey(2000)
    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()
