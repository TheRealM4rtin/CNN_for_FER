import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Concatenate
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from load_img import load_images_parallel  


# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

def create_optimized_cnn_model(input_shape, num_landmarks, num_classes):
    input_img = Input(shape=input_shape)
    x = input_img

    for _ in range(4):
        x = Conv2D(96, (5, 5), activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)

    x = Flatten()(x)

    input_landmarks = Input(shape=(num_landmarks,))
    combined = Concatenate()([x, input_landmarks])
    
    x = Dense(252, activation='relu', kernel_regularizer=l2(0.001))(combined)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    output = Dense(num_classes, activation='softmax', kernel_regularizer=l2(0.001))(x)
    
    model = Model(inputs=[input_img, input_landmarks], outputs=output)
    return model

def train_optimized_cnn(X_train, landmarks_train, y_train, X_val, landmarks_val, y_val, input_shape, num_landmarks, num_classes, batch_size=32, epochs=100):
    model = create_optimized_cnn_model(input_shape, num_landmarks, num_classes)
    
    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
    
    history = model.fit(
        [X_train, landmarks_train], y_train,
        validation_data=([X_val, landmarks_val], y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[reduce_lr, early_stopping]
    )
    
    return model, history

def prepare_data(y, label_encoder=None):
    if label_encoder is None:
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
    else:
        y_encoded = label_encoder.transform(y)
    y_one_hot = to_categorical(y_encoded)
    num_classes = y_one_hot.shape[1]
    return y_one_hot, num_classes, label_encoder

def load_and_prepare_data(csv_path, images_dir, has_labels=True):
    print(f"\nLoading and processing data from {csv_path}...")
    t0 = time.time()
    try:
        df = pd.read_csv(csv_path)
        df = df.drop_duplicates(subset='id', keep='first').set_index('id')
        df = df.dropna()

        print(f"Number of CSV rows: {len(df)}")

        if has_labels:
            y = df['labels'].values
            landmarks = df.drop(['labels'], axis=1)
        else:
            y = None
            landmarks = df

        X_img, image_ids = load_images_parallel(images_dir, landmarks)
        print(f"Image loading completed in {time.time() - t0:.2f} seconds")
        print(f"Number of images loaded: {len(image_ids)}")
        print(f"Shape of X_img: {X_img.shape}")

        df = df.loc[image_ids]
        if has_labels:
            y = df['labels'].values
            landmarks = df.drop(['labels'], axis=1).values
        else:
            landmarks = df.values

        return X_img, landmarks, y, image_ids

    except Exception as e:
        print(f"Error processing data: {str(e)}")
        return None, None, None, None

def main():
    start_time = time.time()

    # Load and prepare training data
    X_train, landmarks_train, y_train, _ = load_and_prepare_data('data/training_set.csv', 'data/training_set', has_labels=True)
    if X_train is None:
        return

    # Load and prepare testing data
    X_test, landmarks_test, _, test_image_ids = load_and_prepare_data('data/testing_data.csv', 'data/testing_img', has_labels=False)
    if X_test is None:
        return

    # Prepare labels for training data
    y_train, num_classes, label_encoder = prepare_data(y_train)

    print(f"Shape of y_train: {y_train.shape}")
    print(f"Number of classes: {num_classes}")

    # Split training data into train and validation sets
    X_train, X_val, y_train, y_val, landmarks_train, landmarks_val = train_test_split(
        X_train, y_train, landmarks_train, test_size=0.2, random_state=42)

    input_shape = X_train.shape[1:]  # Should be (64, 64, 3)
    num_landmarks = landmarks_train.shape[1]

    print("\nTraining the optimized CNN model...")
    model, history = train_optimized_cnn(
        X_train, landmarks_train, y_train,
        X_val, landmarks_val, y_val,
        input_shape, num_landmarks, num_classes,
        batch_size=32, epochs=100
    )

    # Evaluate the model on the validation set
    print("\nEvaluating the model on the validation set...")
    val_loss, val_accuracy = model.evaluate([X_val, landmarks_val], y_val)
    print(f"Validation accuracy: {val_accuracy:.4f}")

    # Make predictions on the test set
    print("\nMaking predictions on the test set...")
    test_predictions = model.predict([X_test, landmarks_test])
    test_predicted_labels = label_encoder.inverse_transform(np.argmax(test_predictions, axis=1))

    # Create a DataFrame with the predictions
    results_df = pd.DataFrame({
        'id': test_image_ids,
        'predicted_label': test_predicted_labels
    })

    # Save the predictions to a CSV file
    results_path = 'test_predictions.csv'
    results_df.to_csv(results_path, index=False)
    print(f"Predictions saved to {results_path}")

    print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()