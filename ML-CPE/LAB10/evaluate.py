from ultralytics import YOLO

def main():
    # โหลดโมเดลที่เราเทรนเสร็จแล้ว
    model = YOLO("runs/detect/runs/garbage_model-3/weights/best.pt")

    # ประเมินผลโมเดลบน Validation Set
    metrics = model.val(
        data="GARBAGE CLASSIFICATION 3.v2-gc1.yolov8/data.yaml",
        split="val",
        imgsz=416,
        device="cpu"
    )

    print("--- Evaluation Results ---")
    print(f"mAP50-95: {metrics.box.map}")
    print(f"mAP50:    {metrics.box.map50}")
    print(f"Precision: {metrics.box.mp}")
    print(f"Recall:    {metrics.box.mr}")

if __name__ == "__main__":
    main()