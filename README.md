🛑 Drowsiness Detection Using MobileNetV2

This project uses **MobileNetV2** and **Computer Vision** to detect signs of driver
drowsiness by classifying whether the driver's eyes are **open** or **closed**.
It uses a transfer learning model trained on grayscale images, converted to RGB
and resized for real-time video analysis.
---
🎯 Project Goal

> Detect if a driver is drowsy by classifying eye states using deep learning in
> real-time video.

---
🧠 MobileNetV2: The Core AI Model
Model Name: MobileNetV2
Framework: TensorFlow / Keras

Purpose: A lightweight deep learning architecture used to classify the driver’s
eye state (open or closed). This classification is crucial for detecting
drowsiness in real-time.
---
📌 Why MobileNetV2?
- Specifically designed for mobile and embedded applications with limited
  computational resources.
- Combines high accuracy with low latency, making it ideal for real-time driver
  monitoring systems.
---
⚙️ Key Architectural Innovations:

1️⃣ Depthwise Separable Convolutions:
   - Reduces the number of parameters and computations.
   - Works in two steps:
       a) Depthwise: A single filter is applied per input channel.
       b) Pointwise: A 1x1 convolution merges the outputs of the depthwise layer.
   - Result: Efficient and fast feature extraction without loss in performance.

2️⃣ Inverted Residuals with Linear Bottlenecks:
   - Uses a special block structure where:
       - Non-linear functions (like ReLU) are applied in high-dimensional space.
       - Linear transformations are used in low-dimensional representations to avoid
         information loss.
   - Ensures performance while keeping the model compact.

     ---
     🛡️ Role in "Safe My Driver" Project:
- MobileNetV2 is used to classify eye states from real-time video frames.
- Enables immediate drowsiness detection and triggers alerts with minimal delay.
- Essential for reducing accidents by intervening when the driver's eyes close
  due to fatigue.
---
✅ Summary:
MobileNetV2 delivers high accuracy, low computational cost, and real-time
inference— making it a core component of our AI-driven drowsiness detection
system.
---
Application in "Safe My Driver":
- Transfer Learning Approach:
  The project utilizes a transfer learning approach, meaning a MobileNetV2 model
  that was pre-trained on a large, general image dataset (like ImageNet) was
  then fine-tuned on a specific dataset of eye images. This allows the model to
  leverage learned features from a vast amount of data while specializing in
  the nuances of eye state detection.
---
🧱 Model Architecture

Base: MobileNetV2 (excluding top layers)

Custom Layers:
  Flatten
  Dense(128, activation='relu')
  Dropout(0.3)
  Dense(1, activation='sigmoid') ← binary classifier
---
📈 Performance

Validation Accuracy: The MobileNetV2 model achieved approximately 95% accuracy
in classifying eye states (`Open_Eyes` vs `Closed_Eyes`).
Real-time Speed: The model is capable of processing video frames in real-time,
enabling immediate detection and response to signs of driver drowsiness.
---
🧾 Classes Detected (2 total)

| Class Name    | Label | Description             |
|---------------|-------|-------------------------|
| Closed_Eyes   | 0     | Eyes closed (drowsy)    |
| Open_Eyes     | 1     | Eyes open (alert)       |
---
📂 Dataset

Datasetname: MRL Eye Dataset is a large-scale dataset of human eye images.
              The images are annotated according to the state of the eye (open
              or closed), presence of glasses, reflections etc.

Classes: Closed_Eyes, Open_Eyes

Structure:
train/
├── Closed_Eyes/
└── Open_Eyes/
each of them contain 2000 image.
---
Preprocessing:
- Grayscale → RGB conversion
- Resize to 224x224
- Normalization: pixel values scaled to [0, 1]
---
🛠️ Tech Stack

- MobileNetV2 (https://keras.io/api/applications/mobilenet_v2/)
- Python
- TensorFlow / Keras
- OpenCV
- Jupyter Notebook
---
📦 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/drowsiness-detection-eyes.git
cd drowsiness-detection-eyes

