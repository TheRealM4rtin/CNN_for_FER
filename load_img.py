import os
import numpy as np
import multiprocessing
import cv2
from math import floor, ceil

def preprocess_image(img, landmarks):
    X_min = floor(min(landmarks[:68]))
    X_max = ceil(max(landmarks[:68]))
    Y_min = floor(min(landmarks[68:]))
    Y_max = ceil(max(landmarks[68:]))

    X_margin = ceil((X_max - X_min) * 0.15)
    Y_margin_minus = ceil((Y_max - Y_min) * 0.4)
    Y_margin_plus = ceil((Y_max - Y_min) * 0.1)

    X_down = max(0, X_min-X_margin)
    X_up = min(X_max+X_margin, img.shape[1]-1)
    Y_down = max(0, Y_min-Y_margin_minus)
    Y_up = min(Y_max+Y_margin_plus, img.shape[0]-1)

    img_cropped = img[Y_down:Y_up, X_down:X_up, :]
    img_resized = cv2.resize(img_cropped, (64, 64))
    return img_resized

def load_image_chunk(args):
    chunk, directory, landmarks_df = args
    images = []
    image_ids = []
    for filename in chunk:
        try:
            img_path = os.path.join(directory, filename)
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    landmarks = landmarks_df.loc[filename].values
                    img_processed = preprocess_image(img, landmarks)
                    img_processed = img_processed / 255.0  # Normalize to [0, 1]
                    images.append(img_processed)
                    image_ids.append(filename)
                else:
                    print(f"Warning: Unable to read image: {img_path}")
            else:
                print(f"Warning: File not found: {img_path}")
        except Exception as e:
            print(f"Error loading image {filename}: {str(e)}")
    return np.array(images), image_ids

def load_images_parallel(directory, landmarks_df):
    filenames = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    num_cores = multiprocessing.cpu_count()
    chunk_size = max(1, len(filenames) // num_cores)
    chunks = [filenames[i:i + chunk_size] for i in range(0, len(filenames), chunk_size)]
    
    with multiprocessing.Pool(num_cores) as pool:
        results = pool.map(load_image_chunk, [(chunk, directory, landmarks_df) for chunk in chunks])
    
    images = np.concatenate([r[0] for r in results if len(r[0]) > 0])
    image_ids = [item for sublist in [r[1] for r in results] for item in sublist]
    return images, image_ids