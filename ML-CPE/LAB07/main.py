import json
import os
import numpy as np

from data_loader import load_data
from preprocessing import to_features
from split_data import split_dataset
from cnn_model import train_model, predict_model
from evaluate import evaluate_model, plot_history
from test_cnn import test_cnn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# แก้ไข Path ให้ชี้มาที่โฟลเดอร์ Data ตามโครงสร้างปัจจุบันในเครื่อง
DATA_PATH = os.path.join(BASE_DIR, "Data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

IMG_SIZE = 48         
TEST_SIZE = 0.2
VAL_SIZE = 0.1
MAX_PER_CLASS = 1000  # จำกัดจำนวนภาพต่อคลาส เพื่อให้รันบน CPU ได้รวดเร็วขึ้น
EPOCHS = 10           # กำหนดจำนวนรอบการเทรนให้เสร็จไวขึ้น
BATCH_SIZE = 64       

def main():
    print("-" * 30)
    print("CNN Facial Expression Recognition (FER2013)")
    print("-" * 30)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Load Dataset
    print("\n[Step 1] Loading FER2013 dataset...")
    images, labels, classes = load_data(DATA_PATH, IMG_SIZE, MAX_PER_CLASS)
    
    if len(images) == 0:
        print(f"\n[Error] ไม่พบข้อมูลรูปภาพในเส้นทาง: {DATA_PATH}")
        print("กรุณาตรวจสอบโครงสร้างโฟลเดอร์ Data ให้ถูกต้อง")
        return

    X = np.stack(images)
    y = np.array(labels)
    print(f"โหลดข้อมูลสำเร็จ: จำนวนภาพทั้งหมด {X.shape[0]} ภาพ, จำนวนคลาส: {len(classes)}")

    # Step 2: Preprocessing
    print("\n[Step 2] Preprocessing data...")
    X_processed = to_features(X)

    # Step 3: Split Dataset
    print("\n[Step 3] Splitting dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        X_processed, y, test_size=TEST_SIZE, val_size=VAL_SIZE
    )

    # Step 4: Train Model
    print("\n[Step 4] Training CNN model...")
    model, history = train_model(
        X_train, y_train, X_val, y_val, 
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE, 
        num_classes=len(classes)
    )

    plot_history(history, OUTPUT_DIR)

    # Step 5: Evaluate Model
    print("\n[Step 5] Evaluating model...")
    evaluate_model(model, X_test, y_test, classes, OUTPUT_DIR)

    # Step 6: Test CNN
    print("\n[Step 6] Running custom test...")
    test_cnn(model, X_test, y_test, classes)

    print("\n🎉 กระบวนการทั้งหมดเสร็จสิ้นสมบูรณ์!")

if __name__ == "__main__":
    main()