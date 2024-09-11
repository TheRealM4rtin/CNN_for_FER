import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import multiprocessing
import time

from cnn_model import create_cnn_model, train_cnn_model
from load_img import load_images_parallel
from cnn_grid_search import optimize_cnn

# Enable TensorFlow Metal
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("TensorFlow Metal enabled")

def prepare_data(y):
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_one_hot = to_categorical(y_encoded)
    num_classes = y_one_hot.shape[1]
    return y_one_hot, num_classes, label_encoder

def main():
    start_time = time.time()

    print("\nLoading and processing CSV data...")
    t0 = time.time()
    try:
        df = pd.read_csv('data/training_set.csv')
        df = df.drop_duplicates(subset='id', keep='first').set_index('id')
        df = df.dropna()

        print(f"Number of CSV rows: {len(df)}")

        y = df['labels'].values
        landmarks = df.drop(['labels'], axis=1)

    except Exception as e:
        print(f"Error processing CSV data: {str(e)}")
        return

    print("Loading image data...")
    t0 = time.time()
    try:
        X_img, image_ids = load_images_parallel('data/training_set', landmarks)
        print(f"Image loading completed in {time.time() - t0:.2f} seconds")
        print(f"Number of images loaded: {len(image_ids)}")
        print(f"Shape of X_img: {X_img.shape}")
    except Exception as e:
        print(f"Error loading images: {str(e)}")
        return

    df = df.loc[image_ids]
    y = df['labels'].values
    landmarks = df.drop(['labels'], axis=1).values

    y, num_classes, label_encoder = prepare_data(y)
    print(f"Shape of y: {y.shape}")
    print(f"Number of classes: {num_classes}")

    X_train, X_test, y_train, y_test, landmarks_train, landmarks_test = train_test_split(
        X_img, y, landmarks, test_size=0.2, random_state=42)

    print("\nPerforming Bayesian Optimization for CNN...")
    t0 = time.time()
    best_params, best_val_accuracy = optimize_cnn(X_train, y_train, landmarks_train)

    final_model = create_cnn_model(
        num_conv_layers=int(round(best_params['num_conv_layers'])),
        filters=(int(round(best_params['filters'])), int(round(best_params['filters']*1.5))),
        kernel_size=(int(round(best_params['kernel_size'])), int(round(best_params['kernel_size']))),
        pool_size=(2, 2),
        dense_units=int(round(best_params['dense_units'])),
        dropout_rate=best_params['dropout_rate'],
        l2_reg=best_params['l2_reg'],
        num_landmarks=landmarks_train.shape[1]
    )

    history = train_cnn_model(
        final_model, 
        X_train, landmarks_train, y_train,
        X_test, landmarks_test, y_test,
        batch_size=32,
        learning_rate=best_params['learning_rate'],
        epochs=100
    )

    test_loss, test_accuracy = final_model.evaluate([X_test, landmarks_test], y_test, verbose=1)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Optimization completed in {time.time() - t0:.2f} seconds")

    print("Best CNN parameters:", best_params)

    print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")

if __name__== '__main__':
    multiprocessing.freeze_support()  # This line is necessary for Windows compatibility
    main()