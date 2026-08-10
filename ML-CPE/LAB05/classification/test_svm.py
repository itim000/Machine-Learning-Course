import json
import joblib
import numpy as np

OUTPUT_DIR = "outputs"

def test_svm(n_samples=5):
    model = joblib.load(f"{OUTPUT_DIR}/svm_model.pkl")
    scaler = joblib.load(f"{OUTPUT_DIR}/scaler.pkl")
    X_test = np.load(f"{OUTPUT_DIR}/X_test.npy")
    y_test = np.load(f"{OUTPUT_DIR}/y_test.npy")
    with open(f"{OUTPUT_DIR}/classes.json") as f:
        classes = json.load(f)

    # สุ่มข้อมูลจาก Test set มาทดลองทำนาย
    index = np.random.choice(len(X_test), n_samples, replace=False)
    X_sample = X_test[index]
    y_sample = y_test[index]

    predictions = model.predict(scaler.transform(X_sample))

    print("\n--- Sample Prediction Results ---")
    for i in range(n_samples):
        pred = classes[predictions[i]]
        true = classes[y_sample[i]]
        correct = predictions[i] == y_sample[i]
        status = "OK" if correct else "WRONG"
        print(f"[{i + 1}] Pred: {pred:<15} True: {true:<15} Result: {status}")

    correct_total = int((predictions == y_sample).sum())
    print(f"\nOverall Correct: {correct_total}/{n_samples}")

if __name__ == "__main__":
    test_svm()