from ultralytics import YOLO
from pathlib import Path

def main():
    # 1. โหลด Pretrained Model
    model = YOLO("yolov8n.pt")

    # 2. เริ่มเทรนโมเดลด้วย CPU (ปรับลด Epochs และ Image Size เพื่อความรวดเร็ว)
    model.train(
        data="GARBAGE CLASSIFICATION 3.v2-gc1.yolov8/data.yaml",
        epochs=3,             # ปรับลดเหลือ 3 รอบ เพื่อให้เทรนเสร็จไวขึ้น (สามารถปรับเพิ่มได้ถ้าเวลาเหลือ)
        imgsz=416,            # ลดขนาดรูปภาพลงจาก 640 เหลือ 416 เพื่อประหยัดเวลา CPU
        batch=8,              
        device="cpu",         # ใช้ CPU ประมวลผล
        project="runs",
        name="garbage_model",
        save=True,
        plots=True
    )

    save_dir = Path(model.trainer.save_dir)
    print("Training completed!")
    print(f"Best Model Saved at: {save_dir / 'weights/best.pt'}")

if __name__ == "__main__":
    main()