import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe components
def initialize_media_pipe():
    mp_face_detection = mp.solutions.face_detection
    mp_face_mesh = mp.solutions.face_mesh
    face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
    return face_detection, face_mesh

# Process a single frame for face detection and landmarks
def process_frame(frame, face_detection, face_mesh):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detections = face_detection.process(rgb_frame).detections
    mesh_results = face_mesh.process(rgb_frame)
    landmarks = mesh_results.multi_face_landmarks[0] if mesh_results.multi_face_landmarks else None
    return detections, landmarks

# Align face using landmarks(Helper for extract_face)
def align_face(image, landmarks, crop=True):
    ih, iw, _ = image.shape
    try:
        if not landmarks:
            raise ValueError("Landmarks not detected")

        left_eye = np.array([int(landmarks.landmark[33].x * iw), int(landmarks.landmark[33].y * ih)])
        right_eye = np.array([int(landmarks.landmark[263].x * iw), int(landmarks.landmark[263].y * ih)])

        delta_y = right_eye[1] - left_eye[1]
        delta_x = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(delta_y, delta_x))

        center = (float((left_eye[0] + right_eye[0]) / 2), float((left_eye[1] + right_eye[1]) / 2))

        if crop:
            expanded_image = cv2.copyMakeBorder(image, ih, ih, iw, iw, cv2.BORDER_REPLICATE)
            expanded_center = (center[0] + iw, center[1] + ih)

            rotation_matrix = cv2.getRotationMatrix2D(expanded_center, angle, 1.0)
            rotated_image = cv2.warpAffine(expanded_image, rotation_matrix, (expanded_image.shape[1], expanded_image.shape[0]))

            cropped_image = rotated_image[ih:ih + ih, iw:iw + iw]
            return cropped_image
        else:
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            aligned_image = cv2.warpAffine(image, rotation_matrix, (iw, ih))
            return aligned_image

    except Exception as e:
        print(f"Error in align_face: {e}")
        return image  # Return the original image if alignment fails

# Calculate square coordinates for cropping with additional scaling and vertical adjustment(Helper for extract_face)
def calculate_square_coordinates(bbox, frame_shape, scale=1.5, vertical_shift=0.0):
    ih, iw, _ = frame_shape
    x, y, w, h = int(bbox.xmin * iw), int(bbox.ymin * ih), int(bbox.width * iw), int(bbox.height * ih)

    # Calculate the center of the bounding box
    cx, cy = x + w // 2, y + h // 2

    # Apply vertical shift to the center
    cy = cy + int(h * vertical_shift)

    # Determine the side length of the square
    side_length = int(max(w, h) * scale)

    # Calculate new square coordinates
    x1 = max(0, cx - side_length // 2)
    y1 = max(0, cy - side_length // 2)
    x2 = min(iw, cx + side_length // 2)
    y2 = min(ih, cy + side_length // 2)

    return x1, y1, x2, y2

# Extract and align face from frame with square cropping
def extract_face(frame, bbox, landmarks=None, scale=1.5, vertical_shift=0.0):
    # Align the frame if landmarks are provided
    aligned_frame = frame
    if landmarks:
        aligned_frame = align_face(frame, landmarks)

    # Calculate square coordinates for cropping
    x1, y1, x2, y2 = calculate_square_coordinates(bbox, aligned_frame.shape, scale=scale, vertical_shift=vertical_shift)

    # Expand the frame if the scale is less than the default to avoid black borders
    if scale < 1.5:
        ih, iw, _ = aligned_frame.shape
        padding_h = int((1.5 - scale) * ih)
        padding_w = int((1.5 - scale) * iw)
        aligned_frame = cv2.copyMakeBorder(aligned_frame, padding_h, padding_h, padding_w, padding_w, cv2.BORDER_REPLICATE)
        x1 += padding_w
        x2 += padding_w
        y1 += padding_h
        y2 += padding_h

    # Extract the face from the aligned frame
    face = aligned_frame[y1:y2, x1:x2].copy()
    return face, (x1, y1, x2 - x1, y2 - y1)