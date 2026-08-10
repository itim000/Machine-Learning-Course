from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def train_svm(X_train, y_train, X_test=None, y_test=None):
    # ทำ StandardScaler สำหรับ Feature
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    kernels = ['linear', 'poly', 'rbf']
    best_acc = -1
    best_model = None
    
    print("\n--- Comparing SVM Kernels ---")
    if X_test is not None and y_test is not None:
        X_test_scaled = scaler.transform(X_test)
        
        for k in kernels:
            model = SVC(kernel=k)
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
            acc = accuracy_score(y_test, preds)
            print(f"Kernel: {k:<8} | Test Accuracy: {acc * 100:.2f}%")
            
            if acc > best_acc:
                best_acc = acc
                best_model = model
    else:
        # กรณีไม่ได้ส่ง Test set มา จะฝึก RBF เป็นค่าเริ่มต้น
        best_model = SVC(kernel='rbf')
        best_model.fit(X_train_scaled, y_train)

    return best_model, scaler

def predict_svm(model, scaler, X_test):
    X_test_scaled = scaler.transform(X_test)
    predictions = model.predict(X_test_scaled)
    return predictions