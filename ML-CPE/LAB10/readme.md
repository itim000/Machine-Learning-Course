# การตรวจจับและจำแนกประเภทขยะ - Object Detection

โปรเจกต์นี้เป็นการปรับใช้สถาปัตยกรรม YOLOv8 สำหรับตรวจจับและจำแนกประเภทขยะ (Garbage Classification) โดยครอบคลุมตั้งแต่ขั้นตอนการฝึกสอนโมเดล (Training), การประเมินผล (Evaluation), การทำนายผลจากรูปภาพ (Image Inference) ไปจนถึงการตรวจจับแบบ Real-time ผ่านกล้องเว็บแคม

## 1. ภาพรวมโปรเจกต์ (Project Overview)

โปรเจกต์นี้ทำการ Fine-tune โมเดล Ultralytics YOLOv8n ด้วย Pretrained Weights บนชุดข้อมูลขยะประเภทต่างๆ

**คุณสมบัติหลัก:**
* ตรวจจับและจำแนกประเภทขยะหลายหมวดหมู่ (เช่น ขยะแก้ว, โลหะ, กระดาษ, พลาสติก)
* ปรับแต่งประสิทธิภาพด้วยเทคนิค Pretrained YOLOv8 Fine-tuning
* ประเมินผลประสิทธิภาพโมเดลด้วยค่า Precision, Recall, mAP50 และ mAP50-95
* รองรับการทำนายผลผ่านรูปภาพและการตรวจจับแบบ Real-time ผ่านกล้องเว็บแคม
* บันทึกข้อมูล Checkpoints และ Log ผลการเทรนอัตโนมัติ

## 2. ชุดข้อมูล (Dataset)

ใช้ชุดข้อมูลขยะที่จัดอยู่ในรูปแบบโครงสร้าง YOLO Annotation Format

| สัดส่วนข้อมูล (Split) | จำนวนรูปภาพ (Images) |
| :--- | :--- |
| ชุดฝึกสอน (Training) | 7,324 |
| ชุดทดสอบเพื่อปรับแต่ง (Validation) | 2,098 |
| ชุดทดสอบวัดผล (Test) | มีให้ในชุดข้อมูล |

**คลาสของวัตถุ (Classes):** `glass`, `metal`, `paper`, `plastic`  
ไฟล์ Label ของ YOLO แต่ละบรรทัดจะประกอบไปด้วย Class ID และพิกัด Bounding Box แบบ Normalized

## 3. โครงสร้างโปรเจกต์ (Project Structure)

```text
LAB10/
├── GARBAGE CLASSIFICATION 3.v2-gc1.yolov8/
│   ├── train/{images,labels}/
│   ├── valid/{images,labels}/
│   ├── test/{images,labels}/
│   └── data.yaml
├── runs/
├── train.py
├── evaluate.py
├── predict.py
├── requirements.txt
└── README.md
```

| สคริปต์ (Script) | วัตถุประสงค์ (Purpose) |
| :--- | :--- |
| `train.py` | เทรนโมเดล YOLOv8 และบันทึกไฟล์น้ำหนัก (Weights) พร้อมกับ Training Logs |
| `evaluate.py` | ประเมินประสิทธิภาพของโมเดลบนรูปภาพชุดทดสอบที่มี Label |
| `predict.py` | ตรวจจับวัตถุในรูปภาพทดสอบแบบอัตโนมัติ |
| `realtime_test.py` | รันการตรวจจับวัตถุแบบ Real-time ผ่านกล้องเว็บแคม |

## 4. การทดสอบตรวจจับวัตถุ (Object Detection)

**การตรวจจับผ่านรูปภาพ (Image Detection)** — รันสคริปต์เพื่อสุ่มตรวจจับรูปภาพ วาดกรอบ Bounding Box พร้อมระบุหมวดหมู่ และบันทึกผลลัพธ์:
```bash
python predict.py
```

**การตรวจจับแบบ Real-time (Real-time Detection)** — เปิดกล้องเว็บแคม (Index 0) เพื่อแสดงผลการตรวจจับแบบเฟรมต่อเฟรม (กด Q เพื่อออกจากโปรแกรม):
```bash
python realtime_test.py
```

*หมายเหตุ: สคริปต์การทำนายผลจะอ้างอิงตำแหน่งไฟล์โมเดล `best.pt` ที่ได้จากการเทรนจริง สำหรับสคริปต์การประเมินผล (Evaluation) จำเป็นต้องใช้ไฟล์ Label ควบคู่กันไป*

## 5. แนวทางการพัฒนาต่อยอด (Future Work)

* ทดลองใช้โมเดล YOLO ที่มีขนาดใหญ่ขึ้น (เช่น YOLOv8m หรือ YOLOv8l)
* ปรับแต่งค่า Hyperparameters เพื่อเพิ่มความแม่นยำ
* เพิ่มประสิทธิภาพการตรวจจับวัตถุขนาดเล็ก (Small-object Detection)
* วิเคราะห์ข้อผิดพลาดแยกตามหมวดหมู่ขยะ (Class-wise Error Analysis)
* วัดค่าความเร็วในการทำนายผล (Inference FPS) และนำไปพัฒนาเป็นเว็บแอปพลิเคชัน

## 6. เอกสารอ้างอิง (References)

* [Roboflow Dataset](https://roboflow.com/)
* [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)