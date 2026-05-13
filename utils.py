import matplotlib.pyplot as plt


CLASS_NAMES = [
    "T-shirt",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle Boot"
]


# =========================================================
# PLOT CURVES
# =========================================================

def plot_curves(losses, accuracies):

    # LOSS CURVE
    plt.figure(figsize=(6, 4))

    plt.plot(losses)

    plt.title("Loss Over Epochs")

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.grid(True)

    plt.show()

    # ACCURACY CURVE
    plt.figure(figsize=(6, 4))

    plt.plot(accuracies)

    plt.title("Accuracy Over Epochs")

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.grid(True)

    plt.show()


# =========================================================
# SHOW PREDICTIONS
# =========================================================

def show_predictions(images, true_labels, pred_labels):

    plt.figure(figsize=(8, 8))

    for i in range(4):

        plt.subplot(2, 2, i + 1)

        plt.imshow(images[i], cmap='gray')

        plt.title(
            f"True: {CLASS_NAMES[true_labels[i]]}\n"
            f"Pred: {CLASS_NAMES[pred_labels[i]]}"
        )

        plt.axis("off")

    plt.tight_layout()

    plt.show()