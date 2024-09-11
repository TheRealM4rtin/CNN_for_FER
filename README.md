# CNN for Facial Emotion Recognition (FER)

## Overview
This repository contains a Convolutional Neural Network (CNN) architecture designed for facial emotion recognition. The model is optimized for performance and includes various layers and configurations to enhance accuracy. The model achieved a validation accuracy of 72.31%

## Clarified CNN Architecture

The detailed architecture for the CNN is based on the optimized parameters:

1. **Input Layer**: 
   - Image input (shape depends on your data, e.g., 64x64x3)

2. **Convolutional Layers**:
   - 4 convolutional layers, each followed by BatchNormalization and MaxPooling
   - Each convolutional layer has:
     - 96 filters
     - 5x5 kernel size
     - ReLU activation
     - L2 regularization with strength 0.001

3. **Flatten Layer**: 
   - Transitions from convolutional to dense layers

4. **Concatenate**: 
   - Combines flattened image features with landmark data

5. **Dense Layer**:
   - 252 units
   - ReLU activation
   - L2 regularization with strength 0.001

6. **Dropout Layer**:
   - Dropout rate of 0.2

7. **Output Layer**:
   - Units equal to the number of classes (e.g., 7 for 7 emotions)
   - Softmax activation
   - L2 regularization with strength 0.001

### Additional Settings:
- **Initial Learning Rate**: 0.0001
- **Optimizer**: Adam (with the specified learning rate)
- **Loss Function**: Categorical crossentropy (assuming multi-class classification)
