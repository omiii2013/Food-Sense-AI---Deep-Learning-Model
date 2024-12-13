# FoodSenseAI: Identifying Food Categories using Deep Learning

## Project Overview
FoodSenseAI is a deep learning project designed to classify food images into **101 distinct categories** using the **EfficientNetB0** model. It leverages the Food-101 dataset, cutting-edge deep learning techniques, and a well-structured training pipeline to achieve high classification accuracy and computational efficiency.

The project implements advanced optimization techniques such as **mixed precision training** and essential callbacks, making it a robust solution for food image classification.

---

## Key Features
1. **Dataset**: 
   - Utilizes the **Food-101 dataset**, which consists of 101,000 images divided into 101 categories.
   - Includes diverse food types and styles for generalization.

2. **Model**: 
   - **EfficientNetB0** pre-trained on ImageNet for feature extraction.
   - Fine-tuned with global average pooling and dense layers for food classification.
   - Mixed precision training to optimize memory usage and speed.

3. **Callbacks for Optimization**:
   - **EarlyStopping**: Stops training if no improvement in performance.
   - **ReduceLROnPlateau**: Dynamically reduces learning rate on performance plateau.
   - **ModelCheckpoint**: Saves the best-performing model during training.
   - **TensorBoard**: Provides visualization tools for monitoring metrics.

4. **Results**:
   - Achieved **84% training accuracy** and **74% validation accuracy** after fine-tuning.
   - Outputs predictions with probabilities for 101 food categories.

---

## Prerequisites
1. **Environment Setup**:
   - Python 3.10+
   - TensorFlow and related libraries (`tensorflow`, `numpy`, `pandas`, etc.)
   - Ensure multiple GPUs support for faster training.

2. **Installing Dependencies**:
   ```bash
   pip install tensorflow pandas numpy matplotlib

## Hardware Resources:

To load the dataset and run the project in Google Colab, you have to change the runtime to "T4" which is a high performance GPU provided by Google.

**Note:** GPU is available for limited time only and might get disconnected everytime when we load dataset as the size of the dataset is too large. So in that case we have to restart the session instead of reconnecting and re-downloading everything again.

Use multiple GPUs to run the code and train the model.
