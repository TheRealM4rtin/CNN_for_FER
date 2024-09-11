import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np

def create_cnn_model(num_conv_layers, filters, kernel_size, pool_size, dense_units, dropout_rate, l2_reg, num_landmarks):
    input_img = Input(shape=(64, 64, 3))
    x = input_img

    if isinstance(filters, int):
        filters = tuple([filters] * num_conv_layers)
    elif len(filters) < num_conv_layers:
        filters = filters + (filters[-1],) * (num_conv_layers - len(filters))
    
    for i in range(num_conv_layers):
        x = Conv2D(filters[i], kernel_size, activation='relu', padding='same', kernel_regularizer=l2(l2_reg))(x)
        x = MaxPooling2D(pool_size)(x)
    
    x = Flatten()(x)

    input_landmarks = Input(shape=(num_landmarks,))
    
    combined = Concatenate()([x, input_landmarks])
    
    x = Dense(dense_units, activation='relu', kernel_regularizer=l2(l2_reg))(combined)
    x = Dropout(dropout_rate)(x)
    output = Dense(7, activation='softmax', kernel_regularizer=l2(l2_reg))(x)
    
    model = Model(inputs=[input_img, input_landmarks], outputs=output)
    return model

def train_cnn_model(model, X_train, landmarks_train, y_train, X_val, landmarks_val, y_val, batch_size, learning_rate, epochs=50):
    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
    
    history = model.fit(
        [X_train, landmarks_train], y_train,
        validation_data=([X_val, landmarks_val], y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
    
    return history

def create_feature_extractor(trained_model):
    feature_extractor = Model(inputs=trained_model.inputs, outputs=trained_model.layers[-2].output)
    return feature_extractor