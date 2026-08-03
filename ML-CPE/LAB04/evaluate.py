import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

os.makedirs('outputs', exist_ok=True)

def evaluate_models(accuracies):
    plt.figure(figsize=(6, 4))
    plt.plot(list(accuracies.keys()), [v * 100 for v in accuracies.values()], marker='o', color='b')
    plt.title('KNN Accuracy for Different k Values')
    plt.xlabel('k Value')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.savefig('outputs/01_k_curve.png')
    plt.close()

def save_confusion_matrix(best_knn, X_test_scaled, y_test):
    y_pred = best_knn.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', 
                xticklabels=best_knn.classes_, yticklabels=best_knn.classes_)
    plt.title('Confusion Matrix (k=5)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('outputs/02_confusion_matrix.png')
    plt.close()
    
    return y_pred

def save_predictions(X_test_raw, y_test, y_pred):
    df_results = X_test_raw.copy()
    df_results['Actual_Species'] = y_test
    df_results['Predicted_Species'] = y_pred
    df_results.to_csv('outputs/predictions.csv', index=False)