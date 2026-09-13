import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

def evaluate_model(model, X_test, y_test):
    # ให้โมเดลลองทำนายผลจากชุดข้อมูลทดสอบ (X_test)
    y_pred = model.predict(X_test)
    
    # คำนวณหาค่าความแม่นยำ (Accuracy) เป็นเปอร์เซ็นต์
    acc = accuracy_score(y_test, y_pred)
    print(f"ความแม่นยำ (Accuracy): {acc * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # สร้างโฟลเดอร์ output อัตโนมัติถ้ายังไม่มีในเครื่อง
    os.makedirs("output", exist_ok=True)
    
    # สร้างและวาดกราฟ Confusion Matrix เพื่อดูผลการทายว่าถูก/ผิดตรงไหนบ้าง
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    
    plt.figure(figsize=(6, 6))
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title("SVM Confusion Matrix")
    plt.savefig("output/confusion_matrix.png", bbox_inches='tight', dpi=300)
    plt.close()
    print("เซฟรูปสำเร็จ -> output/confusion_matrix.png")