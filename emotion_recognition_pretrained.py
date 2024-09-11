import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow warnings

import numpy as np
import pandas as pd
from deepface import DeepFace
from tqdm import tqdm
import cv2
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

# Reuse the load_images_parallel function from emotion_classification.py
def load_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (48, 48))
    return img

def load_images_parallel(image_dir, landmarks_df):
    img_paths = [os.path.join(image_dir, f"{idx}") for idx in landmarks_df.index]
    
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        future_to_path = {executor.submit(load_image, path): path for path in img_paths}
        images = []
        image_ids = []
        
        for future in tqdm(as_completed(future_to_path), total=len(img_paths), desc="Loading images"):
            img_path = future_to_path[future]
            img = future.result()
            if img is not None:
                images.append(img)
                image_ids.append(os.path.splitext(os.path.basename(img_path))[0])
    
    return np.array(images), image_ids

def predict_emotion(img):
    try:
        result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
        return result[0]['dominant_emotion']
    except Exception as e:
        print(f"Error predicting emotion: {str(e)}")
        return None

def main():
    print("Loading and processing CSV data...")
    df = pd.read_csv('data/testing_data.csv')
    df = df.drop_duplicates(subset='id', keep='first').set_index('id')
    df = df.dropna()

    print("Loading image data...")
    X_img, image_ids = load_images_parallel('data/testing_img', df)
    print(f"Number of images loaded: {len(image_ids)}")

    print("Predicting emotions...")
    predictions = []
    for img in tqdm(X_img, desc="Predicting emotions"):
        emotion = predict_emotion(img)
        predictions.append(emotion)

    results_df = pd.DataFrame({'id': image_ids, 'predicted_emotion': predictions})
    results_df.to_csv('pretrained_model_predictions.csv', index=False)
    print("Predictions saved to pretrained_model_predictions.csv")

if __name__ == '__main__':
    multiprocessing.freeze_support()  # This line is necessary for Windows compatibility
    main()
