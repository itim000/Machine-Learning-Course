# ML-08-Deep Convolutional Neural Network (DCNN)

สร้างโมเดล DCNN ตามสถาปัตยกรรม VGG โดยใช้ภาษา Python สำหรับการจำแนกประเภทรูปภาพ (Image Classification) โปรเจกต์นี้ครอบคลุมกระบวนการทั้งหมด ตั้งแต่การโหลดรูปภาพ, การเตรียมข้อมูล (Preprocessing), การแบ่งชุดข้อมูล (Dataset Splitting), การเทรนโมเดล DCNN (VGG), การประเมินผล และการจำแนกประเภทเครื่องแต่งกาย Fashion-MNIST 10 คลาส

## Data (ชุดข้อมูล)

Fashion-MNIST Dataset (Keras Datasets): ชุดข้อมูลภาพเครื่องแต่งกายขาวดำขนาด 28x28 พิกเซล แบ่งออกเป็น 10 คลาส (เสื้อยืด, กางเกง, เสื้อไหมพรม, ชุดเดรส, เสื้อโค้ท, รองเท้าแตะ, เสื้อเชิ้ต, รองเท้าผ้าใบ, กระเป๋า, และรองเท้าบูท)

## Structure (โครงสร้างโปรเจกต์)

LAB08/
├── data_loader.py          # โหลดชุดข้อมูล Fashion-MNIST ผ่าน Keras
├── preprocessing.py       # ปรับขนาดภาพและแปลงรูปแบบสี BGR เป็น RGB
├── split_data.py          # แบ่งชุดข้อมูลออกเป็น Train, Validation และ Test sets
├── vgg_model.py           # สร้าง เทรน บันทึก และทำนายผลด้วยโมเดล VGG
├── evaluate.py            # คำนวณ Accuracy, Classification Report, Confusion Matrix และกราฟประเมินผล
├── test_vgg.py            # ทดสอบสุ่มรูปภาพ 4 รูปมาทำนายผลและแสดงเปรียบเทียบ
├── main.py                # สคริปต์หลักสำหรับควบคุมขั้นตอนการทำงานทั้งหมด (Pipeline)
│
└── outputs/               # โฟลเดอร์เก็บไฟล์ผลลัพธ์ที่สร้างขึ้น
├── labels.npy
├── classes.json
├── features.npy
├── X_train.npy
├── X_val.npy
├── X_test.npy
├── y_train.npy
├── y_val.npy
├── y_test.npy
├── vgg_model.keras
├── history.json
├── confusion_matrix.png
├── training_history.png
└── prediction_sample.png

## Summary (สรุปผลการทำงาน)

โปรเจกต์นี้เป็นการปรับใช้โครงข่ายประสาทเทียมเชิงลึกแบบ VGG (DCNN) สำหรับจำแนกประเภทรูปภาพเครื่องแต่งกาย Fashion-MNIST จำนวน 10 หมวดหมู่ โดยมีการจัดการระบบแบบโมดูลาร์ (Modular Architecture) แยกการทำงานออกเป็นไฟล์ต่าง ๆ อย่างเป็นระเบียบ ตั้งแต่การรับข้อมูล จนถึงการประเมินและทดสอบผลลัพธ์การทำนายภาพจริง