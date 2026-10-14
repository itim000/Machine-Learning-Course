import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
N_SAMPLES = 4

def test_vgg(n_samples=N_SAMPLES):
    model = keras.models.load_model(f"{OUTPUT_DIR}/vgg_model.keras")
    X_test = np.load(f"{OUTPUT_DIR}/X_test.npy")
    y_test = np.load(f"{OUTPUT_DIR}/y_test.npy")
    with open(f"{OUTPUT_DIR}/classes.json") as f:
        classes = json.load(f)

    index = np.random.choice(len(X_test), n_samples, replace=False)
    X_sample = X_test[index]
    y_sample = y_test[index]

    probabilities = model.predict(X_sample, verbose=0)
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)

    cols = int(np.ceil(np.sqrt(n_samples)))
    rows = int(np.ceil(n_samples / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4.5 * rows))
    axes = np.atleast_1d(axes).ravel()

    correct_total = 0
    for i, ax in enumerate(axes):
        if i >= n_samples:
            ax.axis("off")
            continue

        pred = classes[predictions[i]]
        true = classes[y_sample[i]]
        correct = predictions[i] == y_sample[i]
        if correct:
            correct_total += 1
        color = "green" if correct else "red"

        ax.imshow(X_sample[i])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Pred: {pred} ({confidence[i] * 100:.0f}%)\nTrue: {true}", color=color)

    fig.suptitle(f"Sample Predictions ({correct_total}/{n_samples} correct)")
    fig.tight_layout()
    save_path = f"{OUTPUT_DIR}/prediction_sample.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")

if __name__ == "__main__":
    test_vgg()