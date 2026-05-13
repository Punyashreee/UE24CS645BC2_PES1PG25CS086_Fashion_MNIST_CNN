# Convolutional Neural Networks (CNNs)

## Overview

Implementation of a  **Convolutional Neural Network (CNN)**  from scratch for the classification of the **Fashion-MNIST** dataset.

---

# Dataset

Fashion-MNIST dataset provided by TensorFlow/Keras is used for testing the trained CNN model.

## Dataset Details

| Property | Value |
|---|---|
| Total Images | 70,000 |
| Training Images | 60,000 |
| Testing Images | 10,000 |
| Image Size | 28 × 28 |
| Image Type | Grayscale |
| Number of Classes | 10 |

---

# Classes in Fashion-MNIST

| Label | Class Name |
|---|---|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle Boot |

---

# Train/Test Split Used

Although Fashion-MNIST provides 60,000 training images and 10,000 testing images, a subset of the dataset is used for practical experimentation and faster training as the CNN is implemented manually using NumPy and nested loops.

| Dataset | Available | Used |
|---|---|---|
| Training | 60,000 | 5,000 |
| Testing | 10,000 | 1,000 |

---

# Project Structure

```text
UE24CS645BC2_PES1PG25CS086_Fashion_MNIST_CNN/
│
├── tf_env
├── .gitignore
├── activations.py
├── cnn_layers.py
├── cnn.py
├── loss.py
├── utils.py
├── train.py
├── README.md
├── requirements.txt


```

---

# File Description

## activations.py

Contains activation functions:

- ReLU
- ReLU Derivative
- Softmax

---

## cnn_layers.py

Contains implementation of:

- Convolution Layer
- Max Pooling Layer
- Flatten Layer
- Fully Connected Layer

Includes:
- Forward pass (Forward Propagation)
- Backward pass (Backpropagation)

---

## cnn.py

Defines the complete CNN architecture by connecting all the layers.

Architecture:

```text
Input Image
    ↓
Convolution Layer
    ↓
ReLU
    ↓
Max Pooling
    ↓
Flatten
    ↓
Fully Connected Layer
    ↓
Softmax Output
```

---

## loss.py

Contains:

- Cross Entropy Loss
- Accuracy Calculation

---

## utils.py

Contains utility functions for:

- Plotting loss curve
- Plotting accuracy curve
- Displaying predictions

---

## train.py

Main training script.

Responsibilities:
- Load dataset
- Train CNN
- Perform backpropagation
- Evaluate model
- Display graphs and predictions

---

# CNN Architecture

| Layer | Details |
|---|---|
| Convolution Layer | 8 filters of size 3×3 |
| Activation | ReLU |
| Pooling | MaxPool 2×2 |
| Flatten | Converts feature maps into vector |
| Fully Connected | 1352 → 10 neurons |
| Output | Softmax probabilities |

---

# Hyperparameters

| Hyperparameter | Value |
|---|---|
| Epochs | 4 |
| Learning Rate | 0.005 |
| Training Samples | 5000 |
| Testing Samples | 1000 |
| Batch Size | 1 (SGD) |

---

# Steps Per Epoch

Since:
- Training samples = 5000
- Batch size = 1

Steps per epoch:

```text
5000 steps
```

Total training steps:

```text
4 × 5000 = 20000 steps
```

---

# Outputs Generated

The program displays:

## 1. Loss Over Epochs

Shows how prediction error decreases during training.

---

## 2. Accuracy Over Epochs

Shows how model performance improves during training.

---

## 3. Loss Curve

Graph representing:
- Epoch vs Loss

---

## 4. Accuracy Curve

Graph representing:
- Epoch vs Accuracy

---

## 5. Test Accuracy

Displays final accuracy on unseen test data.

Example:

```text
Test Accuracy: 0.8280
```

---

## 6. Prediction Visualization

Displays:
- 4 Fashion-MNIST test images
- True labels
- Predicted labels

Example:

```text
True: Sneaker
Predicted: Sneaker
```

---


# Result Screenshots

## Training Output

<img width="547" height="541" alt="image" src="https://github.com/user-attachments/assets/3ca2d32c-119f-4adb-907e-d35f47cf9766" />
<img width="538" height="461" alt="image" src="https://github.com/user-attachments/assets/a925bd67-8c31-424b-bc4f-bbeed572682b" />
<img width="536" height="487" alt="image" src="https://github.com/user-attachments/assets/b2b06ceb-3fb7-40d8-bd4b-fb2eba708c9c" />
<img width="544" height="509" alt="image" src="https://github.com/user-attachments/assets/9d882212-849b-494d-9308-3029eb21532a" />





---

## Loss Curve

<img width="880" height="603" alt="image" src="https://github.com/user-attachments/assets/47000ac1-2146-46fb-8bf8-b706129a0449" />


---

## Accuracy Curve

<img width="858" height="621" alt="image" src="https://github.com/user-attachments/assets/a206f883-e5a1-44f9-999a-cd379429720d" />


---

## Prediction Results
<img width="547" height="316" alt="image" src="https://github.com/user-attachments/assets/b20784ee-d6f5-4204-bc29-de713b9a9d05" />
<img width="868" height="553" alt="image" src="https://github.com/user-attachments/assets/adf99821-bd6a-4805-967c-03471d1015c6" />



# Technologies Used

- Python
- NumPy
- TensorFlow (only for dataset loading)
- Matplotlib

---

# Installation

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# How to Run

Run the following command:

```bash
python train.py
```

---

# Final Results

| Metric | Value |
|---|---|
| Final Test Accuracy | 82.8% |
| Training Accuracy Progress | 70.08% → 81.96% → 84.54% → 85.94% |
| Training Loss Progress | 0.8612 → 0.5426 → 0.4697 → 0.4270 |

The CNN model successfully learned meaningful image features from the Fashion-MNIST dataset and achieved a final test accuracy of approximately **82.8%** on unseen test data.

The model showed continuous improvement in training accuracy across epochs while simultaneously reducing training loss, indicating successful convergence and effective learning.


---

# Observations

- The training accuracy increased steadily over epochs, showing that the CNN progressively learned better feature representations from the input images.

- The training loss consistently decreased during training, indicating reduced prediction error and improved model optimization.

- The accuracy curve displayed a smooth upward trend, demonstrating stable learning behavior without major fluctuations.

- The loss curve showed a clear downward trajectory, confirming that the model was successfully minimizing classification error over time.

- The final test accuracy of **82.8%** indicates good generalization capability on unseen Fashion-MNIST images.

- The CNN successfully identified important visual patterns such as edges, shapes and textures required for clothing classification.

- Some Fashion-MNIST classes with visually similar appearances (such as Shirt, T-shirt/top and Pullover) are comparatively more challenging to classify accurately.

- The model achieved stable performance by the final epoch, maintaining a good balance between accuracy improvement and loss reduction.

---

# Author

Name: Punyashree G
