# Real-Time-Face-Recognition: Real-Time Face Detection and Recognition

This app is a real-time face recognition system built with Python.  
It captures video from your webcam, detects faces using MediaPipe, aligns them based on facial landmarks, and recognizes the identity by comparing against known reference images.

> Designed with modular architecture, accuracy-focused preprocessing, and user-friendly controls.

---

## Table of Contents
-[Features](#-features)  

-[Technologies Used](#-technologies-used) 

-[Project Structure](#️-project-structure) 

-[Add Reference Faces](#-add-reference-faces)

-[Keyboard Controls](#️-keyboard-controls)  

-[How to Run](#-how-to-run) 

-[Author](#-author)

---

## Features

- Real-time face detection with MediaPipe
- Facial landmark-based alignment for improved accuracy
- Face recognition with similarity match scores
- Live webcam support with adjustable cropping and zoom
- Modular codebase: easy to extend, debug, or adapt

---

## Technologies Used

- **Python 3.8+**
- [OpenCV] - Real-time image and video processing  
- [MediaPipe] – Face detection and mesh landmark extraction  
- [face_recognition] – Facial encoding and comparison  
- [NumPy] – Numerical operations

---

## Project Structure

Real-Time-Face-Recognition/

├── main.py # Main app entry point

├── config.py # Constants and reference face paths

├── face_utils.py # Alignment, detection, cropping

├── recognition.py # Encoding, top match scoring

├── display.py # UI rendering and keyboard controls

├── Faces/ # Reference face images (.jpg)

├── requirements.txt # All Python dependencies

└── README.md # You're reading it!

---

## Add Reference Faces

How to Add:    
1-Place face images inside the Faces/ directory (create if it doesn’t exist):

Real-Time-Face-Recognition/

└── Faces/        
    ├── face1.jpg        
    ├── face2.jpg    
    └── ...       

2-In config.py, map each image to a name:

REFERENCE_IMAGES = {    
    "Alice": os.path.join(FACES_DIR, "face1.jpg"),    
    "Bob": os.path.join(FACES_DIR, "face2.jpg")    
    ...    
}

Tip: Use clear, front-facing images with only one face. 

---

## Keyboard Controls

The app lets you fine-tune cropping and zoom live:

Key	&nbsp;&nbsp;Action    
a	&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Zoom out    
s	&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Zoom in    
d	&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Reset zoom scale    
z	&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Shift crop up    
x	&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Shift crop down    
c	&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Reset vertical shift    
q	&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Quit the app

---

## How to Run

Make sure Python and pip are installed.

Install Dependencies: "pip install -r requirements.txt"

Launch the App: "python main.py"

---

## Author
Built with care by Baran İslam.


