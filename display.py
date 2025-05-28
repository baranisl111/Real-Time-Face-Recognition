import cv2

# Draw bounding box and landmarks
def draw_face_details(frame, bbox, landmarks=None):
    ih, iw, _ = frame.shape
    x, y, w, h = int(bbox.xmin * iw), int(bbox.ymin * ih), int(bbox.width * iw), int(bbox.height * ih)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)# Draw bounding box
    if landmarks:
        # Landmark indices: left eye, right eye, nose tip, mouth corners
        key_points = [33, 263, 1, 61, 291]
        for idx in key_points:
            lx = int(landmarks.landmark[idx].x * iw)
            ly = int(landmarks.landmark[idx].y * ih)
            cv2.circle(frame, (lx, ly), 3, (0, 255, 0), -1)

# Handle keyboard input for adjustments
def handle_keyboard_input(key, scale, vertical_shift, default_scale, default_vertical_shift):
    """
    Controls:
        z - move crop region upward
        x - move crop region downward
        a - decrease crop scale (zoom in)
        s - increase crop scale (zoom out)
        c - reset vertical shift
        d - reset scale
    """
    if key == ord('z'):  # Move up
        vertical_shift -= 0.1
    elif key == ord('x'):  # Move down
        vertical_shift += 0.1
    elif key == ord('a'):  # Decrease scale
        scale = max(0.5, scale - 0.1)
    elif key == ord('s'):  # Increase scale
        scale += 0.1
    elif key == ord('c'):  # Reset vertical shift
        vertical_shift = default_vertical_shift
    elif key == ord('d'):  # Reset scale
        scale = default_scale
    return scale, vertical_shift