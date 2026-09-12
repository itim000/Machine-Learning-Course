import os
import joblib
from sklearn.neural_network import MLPClassifier

def train_model(X_train, y_train, X_val, y_val, output_dir="outputs", epochs=50, batch_size=16):
    # สร้าง Neural Network แบบ MLP
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=epochs,
        batch_size=batch_size,
        random_state=42
    )
    
    print("\nTraining MLP Neural Network...")
    model.fit(X_train, y_train)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(model, os.path.join(output_dir, "heart_nn_model.pkl"))

    # จำลอง history object ให้ evaluate.py ใช้งานต่อได้
    class DummyHistory:
        def __init__(self, loss_curve):
            self.history = {
                'loss': loss_curve,
                'val_loss': loss_curve,
                'accuracy': [model.score(X_train, y_train)] * len(loss_curve),
                'val_accuracy': [model.score(X_val, y_val)] * len(loss_curve)
            }
            
    return model, DummyHistory(model.loss_curve_)

def predict_model(model, X_test):
    return model.predict(X_test)