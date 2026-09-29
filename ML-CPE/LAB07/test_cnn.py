import os
import matplotlib.pyplot as plt
import numpy as np

def test_cnn(model, X_test, y_test, classes):
    if len(X_test) >= 4:
        # สุ่มเลือกรูปภาพจากชุดทดสอบมา 4 รูป
        indices = np.random.choice(len(X_test), 4, replace=False)
        sample_images = X_test[indices]
        true_labels = y_test[indices]
        
        # ทำนายผล
        preds = model.predict(sample_images)
        pred_labels = np.argmax(preds, axis=1)
        pred_probs = np.max(preds, axis=1) * 100
        
        # นับจำนวนที่ทายถูก
        correct_count = np.sum(pred_labels == true_labels)
        
        # สร้างกราฟแสดงผลแบบ 2x2 เหมือนของอาจารย์
        plt.figure(figsize=(7, 7))
        plt.suptitle(f"Prediction: {correct_count}/4 correct", fontsize=14, y=0.95)
        
        for i in range(4):
            plt.subplot(2, 2, i + 1)
            
            # แปลงภาพกลับมาแสดงผล (รองรับทั้งภาพขาวดำและสี)
            img = sample_images[i]
            if img.shape[-1] == 1:
                img = np.squeeze(img, axis=-1)
                plt.imshow(img, cmap='gray')
            else:
                plt.imshow(img)
                
            plt.axis('off')
            
            # เช็คว่าทายถูกไหมเพื่อกำหนดสีข้อความ (เขียว = ถูก, แดง = ผิด)
            is_correct = (pred_labels[i] == true_labels[i])
            color = 'green' if is_correct else 'red'
            
            pred_name = classes[pred_labels[i]]
            true_name = classes[true_labels[i]]
            
            title_text = f"Pred: {pred_name} ({pred_probs[i]:.0f}%)\nTrue: {true_name}"
            plt.title(title_text, color=color, fontsize=11, fontweight='bold')
            
        plt.tight_layout()
        os.makedirs("outputs", exist_ok=True)
        save_path = os.path.join("outputs", "prediction_sample.png")
        plt.savefig(save_path)
        plt.close()
        print(f"บันทึกภาพตัวอย่างการพยากรณ์เรียบร้อยที่ {save_path}")
    else:
        print("ข้อมูลทดสอบมีไม่พอสำหรับแสดง 4 รูป")