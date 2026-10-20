from ultralytics import YOLO

def main():
    # โหลดโมเดล
    model = YOLO("runs/detect/runs/garbage_model-3/weights/best.pt")

    # เปิดกล้อง Webcam (source=0) เพื่อตรวจจับแบบ Real-time
    model.predict(
        source=0,
        show=True,
        conf=0.25
    )

if __name__ == "__main__":
    main()