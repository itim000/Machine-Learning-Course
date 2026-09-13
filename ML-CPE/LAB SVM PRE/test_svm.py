import os
import matplotlib.pyplot as plt

def test_new_messages(model, vectorizer):
    # กำหนดข้อความตัวอย่างใหม่ที่ต้องการป้อนให้ AI ทดสอบ
    new_messages = [
        "Congratulations you have won a free iPhone click here",
        "Hi, are we still meeting for the project tomorrow?",
        "URGENT your account has been locked claim reward now",
        "Can you send me the notes from yesterday class?"
    ]
    
    # แปลงข้อความใหม่ให้เป็นตัวเลขด้วย vectorizer ตัวเดิมที่เคยเทรนมา
    X_new = vectorizer.transform(new_messages)
    
    # ให้โมเดลทำนายผลลัพธ์ของข้อความใหม่
    predictions = model.predict(X_new)
    
    print("\n--- ทดสอบข้อความใหม่ ---")
    for text, pred in zip(new_messages, predictions):
        print(f"ข้อความ: '{text}' --> ผลลัพธ์: {pred.upper()}")
        
    # สร้างโฟลเดอร์ output อัตโนมัติถ้ายังไม่มี
    os.makedirs("output", exist_ok=True)
    
    # สร้างรูปภาพตารางแสดงผลลัพธ์การทดสอบข้อความใหม่เพื่อเก็บไว้ดู
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.axis('off')
    
    table_data = []
    for text, pred in zip(new_messages, predictions):
        status = "SPAM" if pred == 'spam' else "HAM"
        table_data.append([text, status])
        
    table = ax.table(
        cellText=table_data,
        colLabels=["Input Message", "Prediction Result"],
        loc='center',
        cellLoc='left'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    
    plt.title("SVM Spam Detection - Prediction Samples", fontweight='bold', pad=20)
    plt.savefig("output/prediction_sample.png", bbox_inches='tight', dpi=300)
    plt.close()
    print("เซฟรูปสำเร็จ -> output/prediction_sample.png")