# Face Recognition System With DeepFace and Facenet512

This project aims to create a strong, accurate and super simple face recognition system that can be used a source to create many other projects. 

## Why Facenet512 ?

During my experimentations, I've tested out many face recognition models coming from the [DeepFace](https://github.com/serengil/deepface) python library. While we had some models that were much more faster such as OpenFace, ArcFace or SFace I've decided to choose accuracy over speed, leading me to choose Facenet512. Also even this model is quite good, we do have a similary measure that will help you decide if a face is similar to another or not. These similarity measures are the eucludean, the cosine or the euclidean_l2. Those measures have been tested and given a threshold that works pretty well. But in my case, after many experimentation, I've found the best similarity measure and its appropriate threshold that fit pretty well on my dataset.

**Note**: If the model is bad on your use case try to change the similarity measure as well as the threshold. If you want accuracy over speed I strongly recommend to choose the Fcaenet512.

## Scripts Overview

1. `crop_face.py`

This script extracts faces from images using DeepFace and saves them as cropped and resized images in a directory.

    Arguments:

    input_dir: Directory containing input images.
    output_dir: Directory to save cropped faces.
    detector_backend: Backend detector to use (default: "yolov8").



2. `extract_emb.py`

This script preprocesses the cropped faces, extracts facial embeddings using DeepFace, and saves them to a pickle file.

    Arguments:

    input_dir: Directory containing cropped face images.
    output_dir: Directory to save embeddings of the cropped faces.
    emb_file: Filename to save embeddings as a pickle file.
    norm_dir: Directory to save preprocessed (normalized) face images.


3. `face_recognition.py`

This script performs real-time face recognition on a video feed using precomputed embeddings.

## Instructions

Clone the repository

```bash
git clone https://github.com/your_username/your_repository.git
cd Face_recognition_facenet512.git
pip install -r requirements.txt
```

