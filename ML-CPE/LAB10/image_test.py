from ultralytics import YOLO

def main():
    # โหลดโมเดล
    model = YOLO("runs/detect/runs/garbage_model-3/weights/best.pt")

    # ทดสอบตรวจจับรูปภาพในโฟลเดอร์ test
    results = model.predict(
        source="GARBAGE CLASSIFICATION 3.v2-gc1.yolov8/test/images",
        conf=0.25,
        save=True,
        project="runs",
        name="image_test_results"
    )

    print("Predict completed! Check results in 'runs/image_test_results' folder.")

if __name__ == "__main__":
    main()