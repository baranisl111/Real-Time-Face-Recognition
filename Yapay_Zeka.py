import cv2
import mediapipe as mp
import numpy as np
import face_recognition
import os

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

# Align face using landmarks
def align_face(frame, landmarks):
    ih, iw, _ = frame.shape
    try:
        # Get coordinates for left eye (33) and right eye (263)
        left_eye = np.array([int(landmarks.landmark[33].x * iw), int(landmarks.landmark[33].y * ih)])
        right_eye = np.array([int(landmarks.landmark[263].x * iw), int(landmarks.landmark[263].y * ih)])

        # Calculate the angle of rotation
        delta_y = right_eye[1] - left_eye[1]
        delta_x = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(delta_y, delta_x))

        # Get the center point between the eyes and convert to float
        center = (float((left_eye[0] + right_eye[0]) / 2), float((left_eye[1] + right_eye[1]) / 2))

        # Expand the frame to avoid black borders after rotation
        expanded_frame = cv2.copyMakeBorder(frame, ih, ih, iw, iw, cv2.BORDER_REPLICATE)
        expanded_center = (center[0] + iw, center[1] + ih)

        # Get the rotation matrix and apply it to the expanded frame
        rotation_matrix = cv2.getRotationMatrix2D(expanded_center, angle, 1.0)
        rotated_frame = cv2.warpAffine(expanded_frame, rotation_matrix, (expanded_frame.shape[1], expanded_frame.shape[0]))

        # Crop back to the original frame size
        cropped_frame = rotated_frame[ih:ih + ih, iw:iw + iw]
        return cropped_frame
    except Exception as e:
        print(f"Error in align_face: {e}")
        return frame  # Return the original frame if alignment fails

# Calculate square coordinates for cropping with additional scaling and vertical adjustment
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

# Draw bounding box and landmarks
def draw_face_details(frame, bbox, landmarks=None):
    ih, iw, _ = frame.shape
    x, y, w, h = int(bbox.xmin * iw), int(bbox.ymin * ih), int(bbox.width * iw), int(bbox.height * ih)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    if landmarks:
        key_points = [33, 263, 1, 61, 291]
        for idx in key_points:
            lx = int(landmarks.landmark[idx].x * iw)
            ly = int(landmarks.landmark[idx].y * ih)
            cv2.circle(frame, (lx, ly), 3, (0, 255, 0), -1)

# Load reference face encodings with names
def load_reference_encodings_with_names(reference_images):
    reference_encodings = []
    reference_names = []
    for name, path in reference_images.items():
        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)
        if encodings:  # Ensure at least one face encoding is found
            reference_encodings.append(encodings[0])
            reference_names.append(name)
    return reference_encodings, reference_names

# Align face for recognition using landmarks
def preprocess_face_for_recognition(face, landmarks):
    ih, iw, _ = face.shape
    try:
        # Get coordinates for left eye (33) and right eye (263)
        left_eye = np.array([int(landmarks.landmark[33].x * iw), int(landmarks.landmark[33].y * ih)])
        right_eye = np.array([int(landmarks.landmark[263].x * iw), int(landmarks.landmark[263].y * ih)])

        # Calculate the angle of rotation
        delta_y = right_eye[1] - left_eye[1]
        delta_x = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(delta_y, delta_x))

        # Get the center point between the eyes
        center = (float((left_eye[0] + right_eye[0]) / 2), float((left_eye[1] + right_eye[1]) / 2))

        # Get the rotation matrix and apply it to the face
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        aligned_face = cv2.warpAffine(face, rotation_matrix, (iw, ih))
        return aligned_face
    except Exception as e:
        print(f"Error in preprocess_face_for_recognition: {e}")
        return face  # Return the original face if alignment fails

# Recognize face using face_recognition library
def recognize_face_with_library(face, reference_encodings, reference_names, landmarks=None):
    # Align the face for recognition if landmarks are provided
    if landmarks:
        face = preprocess_face_for_recognition(face, landmarks)

    rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face_encodings = face_recognition.face_encodings(rgb_face)
    if face_encodings:  # Ensure at least one encoding is found
        matches = face_recognition.compare_faces(reference_encodings, face_encodings[0])
        face_distances = face_recognition.face_distance(reference_encodings, face_encodings[0])
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            return reference_names[best_match_index]
    return "Unknown"

# Handle keyboard input for adjustments
def handle_keyboard_input(key, scale, vertical_shift, default_scale, default_vertical_shift):
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

# Main function to process video
def process_video():
    face_detection, face_mesh = initialize_media_pipe()
    cap = cv2.VideoCapture(0)

    FIXED_FACE_WIDTH = 300
    FIXED_FACE_HEIGHT = 300

    # Load reference face encodings with names
    base_dir = os.path.dirname(__file__)  # Get the directory of the current script
    reference_images = {
        "Baran": os.path.join(base_dir, "Faces", "face1.jpg"),
        "Fox": os.path.join(base_dir, "Faces", "face2.jpg"),
        "Sakaryaspor": os.path.join(base_dir, "Faces", "face3.jpg"),
        "Inter": os.path.join(base_dir, "Faces", "face4.jpg"),
        "Yusuf": os.path.join(base_dir, "Faces", "face5.jpg"),
        "Bulent": os.path.join(base_dir, "Faces", "face6.jpg")
    }
    reference_encodings, reference_names = load_reference_encodings_with_names(reference_images)

    # Initialize scale and vertical shift values
    default_scale = 1.5
    default_vertical_shift = 0.0
    scale = default_scale
    vertical_shift = default_vertical_shift

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        detections, landmarks = process_frame(frame, face_detection, face_mesh)

        if detections:
            for detection in detections:
                bbox = detection.location_data.relative_bounding_box

                # Extract face with adjustable scale and vertical shift
                face, _ = extract_face(frame, bbox, landmarks, scale=scale, vertical_shift=vertical_shift)
                if face.size > 0:
                    resized_face = cv2.resize(face, (FIXED_FACE_WIDTH, FIXED_FACE_HEIGHT))
                    
                    # Recognize the face with landmarks for better alignment
                    name = recognize_face_with_library(resized_face, reference_encodings, reference_names, landmarks)
                    
                    # Display the recognized name prominently
                    cv2.putText(frame, name, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

                    # Show the separated face in a separate window
                    cv2.imshow("Detected Face", resized_face)

                if landmarks:
                    draw_face_details(frame, bbox, landmarks)

        # Show the main webcam feed
        cv2.imshow("Webcam Feed", frame)

        # Handle keyboard input for adjustments
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Exit on pressing 'q'
            break
        scale, vertical_shift = handle_keyboard_input(key, scale, vertical_shift, default_scale, default_vertical_shift)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_video()





