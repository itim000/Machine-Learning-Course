from data_loader import load_and_preprocess_data
from knn_tf import train_knn
from evaluate import evaluate_models, save_confusion_matrix, save_predictions
from sklearn.metrics import accuracy_score

def main():
    X_train_scaled, X_test_scaled, y_train, y_test, X_test_raw = load_and_preprocess_data('Iris.csv')
    
    k_values = [3, 5, 7]
    accuracies = {}
    
    print("=== ผลการทดลอง KNN ===")
    for k in k_values:
        knn = train_knn(X_train_scaled, y_train, k)
        y_pred = knn.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        accuracies[k] = acc
        print(f"ค่า k = {k} | Test Accuracy = {acc * 100:.2f}%")
        
    evaluate_models(accuracies)
    
    best_knn = train_knn(X_train_scaled, y_train, k=5)
    y_pred_best = save_confusion_matrix(best_knn, X_test_scaled, y_test)
    save_predictions(X_test_raw, y_test, y_pred_best)
    
    print("\n[SUCCESS] บันทึกผลลัพธ์ลงในโฟลเดอร์ outputs/ เรียบร้อยแล้ว!")

if __name__ == '__main__':
    main()