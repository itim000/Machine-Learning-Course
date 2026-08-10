import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate_model(y_test, predictions, classes, save_path=None, X_test=None, sample_save_path=None):
    labels = list(range(len(classes)))
    accuracy = accuracy_score(y_test, predictions)

    print("\n------------ Evaluation ------------------")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    report = classification_report(
        y_test, predictions, labels=labels, target_names=classes, zero_division=0
    )
    print(report)
    
    print("Confusion Matrix:")
    matrix = confusion_matrix(y_test, predictions, labels=labels)
    print(matrix)

    if save_path:
        plot_confusion_matrix(matrix, classes, save_path)
        print(f"Saved: {save_path}")

    if sample_save_path and X_test is not None:
        plot_prediction_samples(X_test, y_test, predictions, classes, sample_save_path)
        print(f"Saved: {sample_save_path}")

    return accuracy

def plot_confusion_matrix(matrix, classes, save_path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(matrix, cmap="Blues")

    ax.set_xticks(np.arange(len(classes)), classes)
    ax.set_yticks(np.arange(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    threshold = matrix.max() / 2
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, matrix[i, j], ha="center", va="center",
                    color="white" if matrix[i, j] > threshold else "black")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

def plot_prediction_samples(X_test, y_test, predictions, classes, save_path):
    """สุ่มข้อมูล Iris มา 4 ตัวอย่างเพื่อแสดงผลการทำนายแบบ Scatter Plot"""
    indices = np.random.choice(len(X_test), min(4, len(X_test)), replace=False)
    fig, axes = plt.subplots(2, 2, figsize=(7, 7))
    
    correct_count = sum(1 for idx in indices if predictions[idx] == y_test[idx])
    fig.suptitle(f"Prediction: {correct_count}/4 correct", fontsize=14)

    for i, ax in enumerate(axes.flat):
        idx = indices[i]
        sample = X_test[idx]
        
        # พล็อตจุดสเปกตรัมฟีเจอร์ Iris (Sepal / Petal)
        ax.bar(["SL", "SW", "PL", "PW"], sample, color='skyblue', edgecolor='black')
        ax.set_ylim(0, max(sample) + 1)

        pred_label = classes[int(predictions[idx])]
        true_label = classes[int(y_test[idx])]

        color = "green" if pred_label == true_label else "red"
        ax.set_title(f"Pred: {pred_label}\nTrue: {true_label}", color=color, fontsize=10)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)