import cv2
import numpy as np
import face_recognition

from face_utils import align_face

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
    
# Recognize face using face_recognition library
def recognize_face_with_library(face, reference_encodings, reference_names, landmarks=None):
    # Align the face for recognition if landmarks are provided
    if landmarks:
        face = align_face(face, landmarks, crop=False)

    rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face_encodings = face_recognition.face_encodings(rgb_face)
    if face_encodings:  # Ensure at least one encoding is found
        face_distances = face_recognition.face_distance(reference_encodings, face_encodings[0])
        best_match_index = np.argmin(face_distances)  # Get the index of the best match
        similarity = (1 - face_distances[best_match_index]) * 100  # Calculate similarity percentage
        if similarity < 60.0:  # Mark as "Unknown" if similarity is below 60%
            return "Unknown", 0.0
        return reference_names[best_match_index], similarity
    return "Unknown", 0.0  