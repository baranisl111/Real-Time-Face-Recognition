import cv2

from config import (
    FIXED_FACE_WIDTH, FIXED_FACE_HEIGHT,
    DEFAULT_SCALE, DEFAULT_VERTICAL_SHIFT,
    REFERENCE_IMAGES
)
from face_utils import initialize_media_pipe, process_frame, extract_face
from recognition import load_reference_encodings_with_names, recognize_face_with_library
from display import  draw_face_details, handle_keyboard_input

# Main function to process video
def process_video():
    face_detection, face_mesh = initialize_media_pipe()
    cap = cv2.VideoCapture(0)

    reference_encodings, reference_names = load_reference_encodings_with_names(REFERENCE_IMAGES)

    scale = DEFAULT_SCALE
    vertical_shift = DEFAULT_VERTICAL_SHIFT

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
                    best_match_name, similarity = recognize_face_with_library(resized_face, reference_encodings, reference_names, landmarks)
                    
                    # Display the best match and its similarity percentage
                    cv2.putText(frame, f"{best_match_name} ({similarity:.2f}%)", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

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
        scale, vertical_shift = handle_keyboard_input(key, scale, vertical_shift, DEFAULT_SCALE, DEFAULT_VERTICAL_SHIFT)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_video()





