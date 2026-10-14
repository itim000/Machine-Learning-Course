import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate_model(y_test, predictions, classes, save_path=None):
    labels = list(range(len(classes)))
    accuracy = accuracy_score(y_test, predictions)

    print("\n------------ Evaluation ------------------")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    report = classification_report(y_test, predictions, labels=labels, target_names=classes, zero_division=0)
    print(report)
    print("Confusion Matrix:")
    matrix = confusion_matrix(y_test, predictions, labels=labels)
    print(matrix)

    if save_path:
        plot_confusion_matrix(matrix, classes, save_path)
        print(f"Saved: {save_path}")

    return accuracy

def plot_confusion_matrix(matrix, classes, save_path):
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(classes, fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (Fashion-MNIST)")

    # วาดตัวเลขลงในแต่ละช่อง
    threshold = matrix.max() / 2
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > threshold else "black", fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

def plot_history(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="validation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Loss")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")