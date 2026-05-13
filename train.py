import numpy as np

from tensorflow.keras.datasets import fashion_mnist

from cnn import CNN

from loss import cross_entropy_loss
from loss import accuracy

from utils import plot_curves
from utils import show_predictions
from utils import CLASS_NAMES


# =========================================================
# LOAD DATASET
# =========================================================

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print("Training Data:", x_train.shape)

print("Testing Data:", x_test.shape)


# =========================================================
# CREATE MODEL
# =========================================================

model = CNN()


# =========================================================
# HYPERPARAMETERS
# =========================================================

epochs = 4

learning_rate = 0.005

train_samples = 5000

test_samples = 1000


# =========================================================
# STORE METRICS
# =========================================================

epoch_losses = []

epoch_accuracies = []


# =========================================================
# TRAINING LOOP
# =========================================================

for epoch in range(epochs):

    print("\n=========================")

    print(f"Epoch {epoch + 1}/{epochs}")

    print("=========================")

    total_loss = 0

    total_accuracy = 0

    for i in range(train_samples):

        image = x_train[i]

        label = y_train[i]

        # FORWARD PASS
        output = model.forward(image)

        # LOSS
        loss = cross_entropy_loss(
            output,
            np.array([label])
        )

        total_loss += loss

        # ACCURACY
        acc = accuracy(
            output,
            np.array([label])
        )

        total_accuracy += acc

        # BACKPROPAGATION
        gradient = output.copy()

        gradient[0, label] -= 1

        # FC BACKWARD
        gradient = model.fc.backward(
            gradient.flatten(),
            learning_rate
        )

        # RESHAPE
        gradient = gradient.reshape(
            8,
            13,
            13
        )

        # MAXPOOL BACKWARD
        gradient = model.pool.backward(
            gradient
        )

        # CONV BACKWARD
        model.conv.backward(
            gradient,
            learning_rate
        )

        # PRINT STATUS
        if i % 500 == 0:

            print(
                f"Step: {i}"
                f" | Loss: {loss:.4f}"
                f" | Accuracy: {acc:.4f}"
            )

    avg_loss = total_loss / train_samples

    avg_accuracy = total_accuracy / train_samples

    epoch_losses.append(avg_loss)

    epoch_accuracies.append(avg_accuracy)

    print("\nEpoch Complete")

    print(f"Average Loss: {avg_loss:.4f}")

    print(f"Average Accuracy: {avg_accuracy:.4f}")


# =========================================================
# TESTING
# =========================================================

print("\n=========================")

print("TESTING MODEL")

print("=========================")

correct = 0

sample_images = []

true_labels = []

predicted_labels = []

for i in range(test_samples):

    image = x_test[i]

    label = y_test[i]

    output = model.forward(image)

    prediction = np.argmax(output)

    if prediction == label:

        correct += 1

    # Store first 4 samples
    if i < 4:

        sample_images.append(image)

        true_labels.append(label)

        predicted_labels.append(prediction)

test_accuracy = correct / test_samples

print(f"\nTest Accuracy: {test_accuracy:.4f}")


# =========================================================
# PLOT CURVES
# =========================================================

plot_curves(
    epoch_losses,
    epoch_accuracies
)


# =========================================================
# SHOW PREDICTIONS
# =========================================================

show_predictions(
    sample_images,
    true_labels,
    predicted_labels
)


# =========================================================
# PRINT TRUE VS PRED
# =========================================================

print("\n=========================")

print("TRUE vs PREDICTED")

print("=========================")

for i in range(4):

    print(
        f"Image {i+1}: "
        f"True = {CLASS_NAMES[true_labels[i]]}"
        f" | Predicted = {CLASS_NAMES[predicted_labels[i]]}"
    )