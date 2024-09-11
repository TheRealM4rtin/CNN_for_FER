import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from scipy.spatial import Delaunay

# Enable TensorFlow Metal
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("TensorFlow Metal enabled")

def load_data(image_dir, csv_path):
    df = pd.read_csv(csv_path)
    df = df.set_index('id')

    images = []
    landmarks = []
    labels = []

    for idx, row in df.iterrows():
        img_path = os.path.join(image_dir, idx)
        if os.path.exists(img_path):
            img = load_img(img_path, target_size=(64, 64))
            img_array = img_to_array(img) / 255.0
            images.append(img_array)
            landmarks.append(row.drop('labels').values)
            labels.append(row['labels'])

    return np.array(images), np.array(landmarks), np.array(labels)

def create_graph_from_landmarks(landmarks):
    print("Landmarks shape:", landmarks.shape)
    
    # Ensure landmarks is 2D array with shape (136, 2)
    if landmarks.shape != (136, 2):
        raise ValueError(f"Expected landmarks shape (136, 2), got {landmarks.shape}")
    
    # Now proceed with the Delaunay triangulation
    tri = Delaunay(landmarks)
    
    # Create adjacency matrix
    num_nodes = landmarks.shape[0]
    adj_matrix = np.zeros((num_nodes, num_nodes))
    
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(3):
                if i != j:
                    adj_matrix[simplex[i], simplex[j]] = 1
    
    return adj_matrix

class GraphConvLayer(layers.Layer):
    def __init__(self, units, activation=None):
        super(GraphConvLayer, self).__init__()
        self.units = units
        self.activation = activation

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='glorot_uniform',
                                 name='kernel')
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='zeros',
                                 name='bias')

    def call(self, inputs, adj):
        x = tf.matmul(adj, inputs)
        output = tf.matmul(x, self.w) + self.b
        if self.activation is not None:
            output = self.activation(output)
        return output

def create_dgnn_model(input_shape, num_landmarks, num_classes):
    image_input = layers.Input(shape=input_shape)
    landmark_input = layers.Input(shape=(num_landmarks, 2))
    adj_matrix_input = layers.Input(shape=(num_landmarks, num_landmarks))

    # CNN for image processing
    x = layers.Conv2D(32, (3, 3), activation='relu')(image_input)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)

    # Graph Convolutional layers for landmarks
    g = GraphConvLayer(64, activation='relu')(landmark_input, adj_matrix_input)
    g = GraphConvLayer(32, activation='relu')(g, adj_matrix_input)
    g = layers.Flatten()(g)

    # Combine CNN and GCN outputs
    combined = layers.concatenate([x, g])
    output = layers.Dense(64, activation='relu')(combined)
    output = layers.Dense(num_classes, activation='softmax')(output)

    model = Model(inputs=[image_input, landmark_input, adj_matrix_input], outputs=output)
    return model

def main():
    # Load and preprocess data
    images, landmarks, labels = load_data('data/training_set', 'data/training_set.csv')

    # Encode labels
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    labels_onehot = to_categorical(labels_encoded)

    # Reshape landmarks to (136, 2)
    landmarks = landmarks.reshape(-1, 136, 2)
    
    # Create graph adjacency matrix for all samples
    adj_matrices = np.array([create_graph_from_landmarks(lm) for lm in landmarks])
    
    # Split the data
    X_img_train, X_img_test, X_lm_train, X_lm_test, y_train, y_test, adj_train, adj_test = train_test_split(
        images, landmarks, labels_onehot, adj_matrices, test_size=0.2, random_state=42)

    # Create and compile the model
    model = create_dgnn_model(input_shape=(64, 64, 3), num_landmarks=landmarks.shape[1], num_classes=len(le.classes_))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model (use adj_train directly)
    model.fit([X_img_train, X_lm_train, adj_train], y_train, 
              epochs=10, batch_size=32, validation_split=0.2, verbose=1)

    # Evaluate the model (use adj_test directly)
    test_loss, test_accuracy = model.evaluate([X_img_test, X_lm_test, adj_test], y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")

if __name__ == '__main__':
    main()