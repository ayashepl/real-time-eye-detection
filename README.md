```yaml
# Drowsiness Detection Using MobileNetV2
project:
  title: Drowsiness Detection Using MobileNetV2
  description: |
    This project uses MobileNetV2 and Computer Vision to detect signs of driver drowsiness by classifying 
    whether the driver's eyes are open or closed. It uses a transfer learning model trained on grayscale images, 
    converted to RGB and resized for real-time video analysis.

# Project Goal
goal:
  objective: Detect if a driver is drowsy by classifying eye states using deep learning in real-time video.

# MobileNetV2: The Core AI Model
model:
  name: MobileNetV2
  framework: TensorFlow / Keras
  purpose: |
    A lightweight deep learning architecture used to classify the driver’s eye state (open or closed). 
    This classification is crucial for detecting drowsiness in real-time.
  why_chosen:
    - Specifically designed for mobile and embedded applications with limited computational resources.
    - Combines high accuracy with low latency, making it ideal for real-time driver monitoring systems.
  architectural_innovations:
    depthwise_separable_convolutions:
      description: |
        Reduces the number of parameters and computations.
        Works in two steps:
          a) Depthwise: A single filter is applied per input channel.
          b) Pointwise: A 1x1 convolution merges the outputs of the depthwise layer.
        Result: Efficient and fast feature extraction without loss in performance.
    inverted_residuals_linear_bottlenecks:
      description: |
        Uses a special block structure where:
          - Non-linear functions (like ReLU) are applied in high-dimensional space.
          - Linear transformations are used in low-dimensional representations to avoid information loss.
        Ensures performance while keeping the model compact.
  role_in_project:
    - MobileNetV2 is used to classify eye states from real-time video frames.
    - Enables immediate drowsiness detection and triggers alerts with minimal delay.
    - Essential for reducing accidents by intervening when the driver's eyes close due to fatigue.
  summary: |
    MobileNetV2 delivers high accuracy, low computational cost, and real-time inference—
    making it a core component of our AI-driven drowsiness detection system.

# Model Summary
model_summary:
  model: MobileNetV2
  framework: TensorFlow / Keras
  training_platform: Jupyter Notebook (with GPU support recommended)
  input_shape: 224x224 RGB

# Performance
performance:
  validation_accuracy: |
    The MobileNetV2 model achieved approximately 95% accuracy in classifying eye states (Open_Eyes vs Closed_Eyes).
  real_time_speed: |
    The model is capable of processing video frames in real-time, enabling immediate detection 
    and response to signs of driver drowsiness.

# Classes Detected
classes:
  total: 2
  list:
    - name: Closed_Eyes
      label: 0
      description: Eyes closed (drowsy)
    - name: Open_Eyes
      label: 1
      description: Eyes open (alert)

# Tech Stack
tech_stack:
  - MobileNetV2
  - Python
  - TensorFlow / Keras
  - OpenCV
  - Jupyter Notebook

# Dataset
dataset:
  total_images: Estimate manually from folder
  split: All images in a folder-based structure
  classes:
    - Closed_Eyes
    - Open_Eyes
  structure: |
    train/
    ├── Closed_Eyes/
    └── Open_Eyes/
  preprocessing:
    - Grayscale → RGB conversion
    - Resize to 224x224
    - Normalization: pixel values scaled to [0, 1]

# How to Run the Project
setup_instructions:
  clone_repository:
    step: 1
    description: Clone the repository
    command: |
      git clone https://github.com/YOUR_USERNAME/drowsiness-detection-eyes.git
      cd drowsiness-detection-eyes
  install_dependencies:
    step: 2
    description: Install dependencies
    command: pip install tensorflow opencv-python matplotlib
  run_notebook:
    step: 3
    description: Run the notebook
    command: jupyter notebook drowsiness_detection.ipynb

# Model Architecture
architecture:
  base: MobileNetV2 (excluding top layers)
  custom_layers:
    - Flatten
    - Dense(128, activation='relu')
    - Dropout(0.3)
    - Dense(1, activation='sigmoid') ← binary classifier

# Training & Evaluation
training_evaluation:
  training: Performed on the preprocessed dataset using MobileNetV2 with transfer learning
  evaluation: Achieved approximately 95% validation accuracy for reliable real-time eye state classification
```
