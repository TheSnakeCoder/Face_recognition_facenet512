# Face Recognition System With DeepFace and Facenet512

["defew"](https://p7.hiclipart.com/preview/348/200/954/mark-zuckerberg-deepface-facebook-facial-recognition-system-deep-learning-landmarks.jpg)

This project aims to create a robust, accurate, and simple face recognition system that can serve as a foundation for various other projects.

## Why Facenet512?

During my experiments, I tested many face recognition models from the [DeepFace](https://github.com/serengil/deepface) Python library. While some models, such as OpenFace, ArcFace, and SFace, were faster, I prioritized accuracy over speed, leading me to choose Facenet512. This model provides excellent accuracy and includes similarity measures to determine if one face is similar to another. These measures include Euclidean, Cosine, and Euclidean_L2 distances. After extensive experimentation, I found the best similarity measure and threshold for my dataset.

**Note**: If the model does not perform well for your use case, consider changing the similarity measure and threshold. For those prioritizing accuracy over speed, I strongly recommend Facenet512.

## Scripts Overview

### 1. `crop_face.py`

This script extracts faces from images using DeepFace and saves them as cropped and resized images in a directory.

- **Arguments**:
  - `input_dir`: Directory containing input images.
  - `output_dir`: Directory to save cropped faces.
  - `detector_backend`: Backend detector to use (default: "yolov8").

### 2. `extract_emb.py`

This script preprocesses the cropped faces, extracts facial embeddings using DeepFace, and saves them to a pickle file.

- **Arguments**:
  - `input_dir`: Directory containing cropped face images.
  - `output_dir`: Directory to save embeddings of the cropped faces.
  - `emb_file`: Filename to save embeddings as a pickle file.
  - `norm_dir`: Directory to save preprocessed (normalized) face images.

### 3. `face_recognition.py`

This script performs real-time face recognition on a video feed using precomputed embeddings.

## Instructions

Clone the repository

```bash
git clone https://github.com/TheSnakeCoder/Face_recognition_facenet512.git
cd Face_recognition_facenet512.git
pip install -r requirements.txt
```

* Create a directory "data" where you can store images of people you want to recognize (make sure the faces are well visible and are looking at the camera.  If the faces captured are bad qualities this could affect the model performence)
* run the `crop_face.py` to crop the faces we want to recognize
* run the `extract_embs.py` to extract the embeddings of the cropped faces
* run the `face_recognition.py`
