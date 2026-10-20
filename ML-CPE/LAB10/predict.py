from ultralytics import YOLO

# 1. โหลดโมเดลที่เราเพิ่งเทรนเสร็จ (เช็ก path โฟลเดอร์ให้ตรงกับที่มีในเครื่อง)
model = YOLO("runs/detect/runs/garbage_model-3/weights/best.pt")

# 2. ทำนายผลรูปภาพในโฟลเดอร์ test
results = model.predict(
    source="GARBAGE CLASSIFICATION 3.v2-gc1.yolov8/test/images",  # ชี้ไปที่โฟลเดอร์รูปทดสอบ
    conf=0.25,  # เกณฑ์ความมั่นใจขั้นต่ำ 25%
    save=True   # บันทึกรูปผลลัพธ์
)

print("Prediction completed! Check results in 'runs/detect/predict' folder.")