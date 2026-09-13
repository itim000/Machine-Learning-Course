# ระบบตรวจจับอีเมลสแปมด้วย Support Vector Machine (SVM)

สร้างไปป์ไลน์การตรวจจับอีเมลสแปมแบบโมดูลาร์ด้วยภาษา Python ประกอบด้วยการโหลดข้อมูลข้อความ การประมวลผลล่วงหน้าด้วย TF-IDF การแบ่งชุดข้อมูล การฝึกสอนโมเดล SVM การประเมินผล และการทำนายผลข้อความใหม่

# ข้อมูล (Data)

ชุดข้อมูล Kaggle SMS (`spam.csv`): https://www.kaggle.com/code/ishansoni/sms-spam-collection-dataset/input

# โครงสร้างโปรเจกต์ (Structure)

```text
LAB-SVM-PRE/
│
├── spam.csv
│
├── data_load.py
├── preprocess.py
├── split_data.py
├── svm_model.py
├── evaluate.py
├── test_svm.py
├── main.py
│
└── output/
    ├── confusion_matrix.png
    └── prediction_sample.png

สรุป (Summary)
โปรเจกต์นี้ใช้โมเดล Support Vector Machine (SVM) แบบ Linear Kernel สำหรับการจำแนกอีเมลสแปมและข้อความปกติ  โดยโหลดข้อความจากไฟล์ CSV ในเครื่อง แปลงเป็นเวกเตอร์ตัวเลขด้วย TF-IDF (Term Frequency-Inverse Document Frequency) และแบ่งข้อมูลเป็นชุดฝึกสอนกับชุดทดสอบ โมเดลที่ฝึกสอนแล้วจะถูกประเมินผลด้วยค่าความแม่นยำ (Accuracy) รายงานการจำแนกประเภท (Precision, Recall, F1-score) และ Confusion Matrix พร้อมทั้งทดสอบทำนายผลกับข้อความใหม่และบันทึกภาพผลลัพธ์อัตโนมัติ