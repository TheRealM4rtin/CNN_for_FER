import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from scipy.spatial import Delaunay
import time
from load_img import load_images_parallel
from skopt import BayesSearchCV
from skopt.space import Real
from sklearn.base import BaseEstimator, ClassifierMixin
from functools import partial


class GraphConvLayer(layers.Layer):
    def __init__(self, units, activation=None, use_bias=True, kernel_regularizer=None):
        super(GraphConvLayer, self).__init__()
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='glorot_uniform',
                                 name='kernel',
                                 regularizer=self.kernel_regularizer)
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

def create_graph_from_landmarks(landmarks):
    tri = Delaunay(landmarks)
    num_landmarks = landmarks.shape[0]
    adj_matrix = np.zeros((num_landmarks, num_landmarks))
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i+1, 3):
                adj_matrix[simplex[i], simplex[j]] = 1
                adj_matrix[simplex[j], simplex[i]] = 1
    master_node = num_landmarks // 2
    adj_matrix[master_node, :] = 1
    adj_matrix[:, master_node] = 1
    adj_matrix[master_node, master_node] = 0
    return adj_matrix

def prepare_data(y):
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_one_hot = to_categorical(y_encoded)
    num_classes = y_one_hot.shape[1]
    return y_one_hot, num_classes, label_encoder

class FacialEmotionRecognitionEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self, learning_rate=0.001, l2_reg=0.01, dropout_rate=0.5):
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.model = None

    def create_combined_model(self, image_shape, num_landmarks, num_classes):
        image_input = layers.Input(shape=image_shape)
        x = layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(self.l2_reg))(image_input)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(self.l2_reg))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(self.l2_reg))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(self.l2_reg))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout_rate)(x)

        landmark_input = layers.Input(shape=(num_landmarks, 2))
        adj_matrix_input = layers.Input(shape=(num_landmarks, num_landmarks))

        g = GraphConvLayer(128, activation='relu', kernel_regularizer=regularizers.l2(self.l2_reg))(landmark_input, adj_matrix_input)
        g = layers.BatchNormalization()(g)
        g = GraphConvLayer(64, activation='relu', kernel_regularizer=regularizers.l2(self.l2_reg))(g, adj_matrix_input)
        g = layers.BatchNormalization()(g)
        g = layers.Flatten()(g)
        g = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(self.l2_reg))(g)
        g = layers.BatchNormalization()(g)
        g = layers.Dropout(self.dropout_rate)(g)

        combined = layers.concatenate([x, g])
        combined = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(self.l2_reg))(combined)
        combined = layers.BatchNormalization()(combined)
        combined = layers.Dropout(self.dropout_rate)(combined)
        output = layers.Dense(num_classes, activation='softmax')(combined)

        model = Model(inputs=[image_input, landmark_input, adj_matrix_input], outputs=output)
        return model

    def fit(self, X, y):
        X_img, X_lm, adj = X
        image_shape = X_img.shape[1:]
        num_landmarks = X_lm.shape[1]
        num_classes = y.shape[1]

        self.model = self.create_combined_model(image_shape, num_landmarks, num_classes)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        early_stopping = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        self.model.fit(
            [X_img, X_lm, adj], y,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )
        return self

    def predict(self, X):
        X_img, X_lm, adj = X
        return self.model.predict([X_img, X_lm, adj])

    def score(self, X, y):
        X_img, X_lm, adj = X
        _, accuracy = self.model.evaluate([X_img, X_lm, adj], y, verbose=0)
        return accuracy

def make_estimator(X, estimator_class, **params):
    return estimator_class(**params)

class CustomOptimizer:
    def __init__(self, estimator_class, search_spaces, n_iter=50, random_state=None):
        self.estimator_class = estimator_class
        self.search_spaces = search_spaces
        self.n_iter = n_iter
        self.random_state = random_state
        self.best_params_ = None
        self.best_score_ = None
        self.best_estimator_ = None

    def fit(self, X, y):
        estimator_partial = partial(make_estimator, X, self.estimator_class)

        bayes_search = BayesSearchCV(
            estimator_partial,
            self.search_spaces,
            n_iter=self.n_iter,
            cv=[(slice(None), slice(None))],  # No CV, use all data
            n_jobs=1,
            verbose=2
        )

        bayes_search.fit(X[0], y)  # Use X[0] as a dummy input

        self.best_params_ = bayes_search.best_params_
        self.best_score_ = bayes_search.best_score_
        self.best_estimator_ = self.estimator_class(**self.best_params_)
        self.best_estimator_.fit(X, y)

        return self

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

    # Prepare the input data
    X_train = [X_img_train, X_lm_train, adj_train]
    X_test = [X_img_test, X_lm_test, adj_test]

    # Define the search space
    search_spaces = {
        'learning_rate': Real(1e-4, 1e-2, prior='log-uniform'),
        'l2_reg': Real(1e-6, 1e-3, prior='log-uniform'),
        'dropout_rate': Real(0.1, 0.5)
    }

    # Create and fit the custom optimizer
    optimizer = CustomOptimizer(
        estimator_class=FacialEmotionRecognitionEstimator,
        search_spaces=search_spaces,
        n_iter=20
    )
    optimizer.fit(X_train, y_train)

    # Print the best parameters and score
    print("Best parameters found: ", optimizer.best_params_)
    print("Best score: ", optimizer.best_score_)

    # Get the best model
    best_model = optimizer.best_estimator_

    # Evaluate the best model on the test set
    test_accuracy = best_model.score(X_test, y_test)
    print(f"Test accuracy with best model: {test_accuracy:.4f}")

    print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")

if __name__ == '__main__':
    main()