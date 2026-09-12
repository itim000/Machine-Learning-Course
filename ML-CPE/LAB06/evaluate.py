import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from nn_model import predict_model

def evaluate_and_save(model, X_test, y_test, history, output_dir='outputs'):
    os.makedirs(output_dir, exist_ok=True)
    
    y_pred = predict_model(model, X_test)
    acc = accuracy_score(y_test, y_pred)
    
    # 1. พล็อตกราฟ Accuracy & Loss
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()

    # 2. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Heart Disease', 'Heart Disease'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix (Test Acc: {acc:.4f})')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    return acc

def plot_prediction_samples(model, X_test, y_test, output_dir='outputs', num_samples=4):
    y_pred = predict_model(model, X_test[:num_samples])
    
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    axes = axes.ravel()
    
    target_names = ['No Heart Disease', 'Heart Disease']
    
    for i in range(num_samples):
        pred_label = y_pred[i]
        true_label = y_test[i]
        color = 'green' if pred_label == true_label else 'red'
        
        axes[i].text(0.1, 0.5, f"Patient #{i+1}\nTrue: {target_names[true_label]}\nPred: {target_names[pred_label]}", 
                     fontsize=12, color=color, weight='bold')
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_sample.png'))
    plt.close()