from deepface import DeepFace
import os
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt


def crop(input_dir, output_dir, detector_backend="yolov8"):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for img_file in tqdm(os.listdir(input_dir)):
        img_path = os.path.join(input_dir, img_file)
        img_name = img_file.split(".")[0]

        face = DeepFace.extract_faces(
            img_path,
            detector_backend=detector_backend,
            enforce_detection=True,
            grayscale=True,
            target_size=None,
        )[0]["face"]
        
        face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
        face = cv2.resize(face, (224, 224))
        plt.imsave(f"{output_dir}/{img_name}.jpg", face)


crop("./data", "./cropped_faces")
