import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow warnings

from cnn_model import create_cnn_model, train_cnn_model
import numpy as np
from sklearn.model_selection import KFold
from bayes_opt import BayesianOptimization

def cnn_bayesian_optimization(X_train, y_train, landmarks_train):
    def cnn_objective(num_conv_layers, filters, kernel_size, dense_units, dropout_rate, l2_reg, learning_rate):
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []

        for train_index, val_index in kfold.split(X_train):
            X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
            y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
            landmarks_train_fold, landmarks_val_fold = landmarks_train[train_index], landmarks_train[val_index]

            cnn_model = create_cnn_model(
                num_conv_layers=int(num_conv_layers),
                filters=(int(filters), int(filters*2)),
                kernel_size=(int(kernel_size), int(kernel_size)),
                pool_size=(2, 2),
                dense_units=int(dense_units),
                dropout_rate=dropout_rate,
                l2_reg=l2_reg,
                num_landmarks=landmarks_train.shape[1]
            )
            
            history = train_cnn_model(
                cnn_model, 
                X_train_fold, landmarks_train_fold, y_train_fold,
                X_val_fold, landmarks_val_fold, y_val_fold,
                batch_size=32,
                learning_rate=learning_rate,
                epochs=30  # Increased epochs for better convergence
            )
            
            cv_scores.append(max(history.history['val_accuracy']))
        
        return np.mean(cv_scores)

    # Define the parameter space
    pbounds = {
        'num_conv_layers': (2, 4),
        'filters': (32, 128),
        'kernel_size': (3, 5),
        'dense_units': (64, 256),
        'dropout_rate': (0.2, 0.5),
        'l2_reg': (1e-6, 1e-3),
        'learning_rate': (1e-4, 1e-2)
    }

    optimizer = BayesianOptimization(
        f=cnn_objective,
        pbounds=pbounds,
        random_state=1,
    )

    optimizer.maximize(
        init_points=5,
        n_iter=25,
    )

    return optimizer.max['params'], optimizer.max['target']

def optimize_cnn(X_train, y_train, landmarks_train):
    best_params, best_val_accuracy = cnn_bayesian_optimization(
        X_train, y_train, landmarks_train
    )

    print("Best CNN parameters:", best_params)
    print("Best validation accuracy:", best_val_accuracy)

    return best_params, best_val_accuracy