import tensorflow as tf
from tensorflow.keras import layers, Model, activations
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from scipy.spatial import Delaunay
import time
from load_img import load_images_parallel

def create_graph_from_landmarks(landmarks):
    """
    Create an adjacency matrix using Delaunay triangulation.
    
    :param landmarks: numpy array of shape (num_landmarks, 2)
    :return: adjacency matrix of shape (num_landmarks, num_landmarks)
    """
    # Perform Delaunay triangulation
    tri = Delaunay(landmarks)
    
    # Create adjacency matrix
    num_landmarks = landmarks.shape[0]
    adj_matrix = np.zeros((num_landmarks, num_landmarks))
    
    # Fill the adjacency matrix based on the Delaunay triangulation
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i+1, 3):
                adj_matrix[simplex[i], simplex[j]] = 1
                adj_matrix[simplex[j], simplex[i]] = 1  # Ensure symmetry
    
    # Add connections to the "master" node (assuming it's the center point, e.g., nose)
    master_node = num_landmarks // 2  # This assumes the center point is in the middle of the landmark list
    adj_matrix[master_node, :] = 1
    adj_matrix[:, master_node] = 1
    adj_matrix[master_node, master_node] = 0  # Remove self-connection
    
    return adj_matrix

def prepare_data(y):
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_one_hot = to_categorical(y_encoded)
    num_classes = y_one_hot.shape[1]
    return y_one_hot, num_classes, label_encoder

class GraphConvLayer(layers.Layer):
    def __init__(self, units, activation=None, use_bias=True):
        super(GraphConvLayer, self).__init__()
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='glorot_uniform',
                                 name='kernel')
        if self.use_bias:
            self.b = self.add_weight(shape=(self.units,),
                                     initializer='zeros',
                                     name='bias')

    def call(self, inputs, adj):
        x = tf.matmul(adj, inputs)
        output = tf.matmul(x, self.w)
        if self.use_bias:
            output += self.b
        if self.activation is not None:
            output = self.activation(output)
        return output

def create_combined_model(image_shape, num_landmarks, num_classes):
    # Image input branch
    image_input = layers.Input(shape=image_shape)
    x = layers.Conv2D(64, (3, 3), padding='same')(image_input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(256, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    # Landmark input branch
    landmark_input = layers.Input(shape=(num_landmarks, 2))
    adj_matrix_input = layers.Input(shape=(num_landmarks, num_landmarks))

    # Graph Convolutional layers for landmarks
    g = GraphConvLayer(128, activation='relu')(landmark_input, adj_matrix_input)
    g = layers.BatchNormalization()(g)
    g = GraphConvLayer(64, activation='relu')(g, adj_matrix_input)
    g = layers.BatchNormalization()(g)
    g = layers.Flatten()(g)
    g = layers.Dense(256, activation='relu')(g)
    g = layers.BatchNormalization()(g)
    g = layers.Dropout(0.5)(g)

    # Combine CNN and GCN outputs
    combined = layers.concatenate([x, g])
    combined = layers.Dense(256, activation='relu')(combined)
    combined = layers.BatchNormalization()(combined)
    combined = layers.Dropout(0.5)(combined)
    output = layers.Dense(num_classes, activation='softmax')(combined)

    model = Model(inputs=[image_input, landmark_input, adj_matrix_input], outputs=output)
    return model

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

    # Reshape landmarks to (num_samples, num_landmarks, 2)
    num_landmarks = landmarks.shape[1] // 2
    landmarks = landmarks.reshape(-1, num_landmarks, 2)

    # Normalize landmark data
    landmarks = (landmarks - np.min(landmarks)) / (np.max(landmarks) - np.min(landmarks))

    # Create adjacency matrices using Delaunay triangulation
    print("Creating adjacency matrices...")
    t0 = time.time()
    adj_matrices = np.array([create_graph_from_landmarks(lm) for lm in landmarks])
    print(f"Adjacency matrix creation completed in {time.time() - t0:.2f} seconds")

    # Split the data
    X_img_train, X_img_test, X_lm_train, X_lm_test, y_train, y_test, adj_train, adj_test = train_test_split(
        X_img, landmarks, y, adj_matrices, test_size=0.2, random_state=42)

    # Create and compile the model
    image_shape = X_img.shape[1:]  # Assuming X_img shape is (num_samples, height, width, channels)
    model = create_combined_model(image_shape, num_landmarks, num_classes)

    # Use a lower learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Implement early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

    # Train the model
    history = model.fit(
        [X_img_train, X_lm_train, adj_train], y_train, 
        epochs=100,  # Increase number of epochs
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate([X_img_test, X_lm_test, adj_test], y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")

    print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")
if __name__ == '__main__':
    main()