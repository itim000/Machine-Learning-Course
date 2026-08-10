import os
import json
import pandas as pd
import numpy as np

def load_data(data_path="../Iris.csv", **kwargs):
    """อ่านข้อมูลจาก Iris.csv แปลง Feature และ Label ลงในระบบ"""
    df = pd.read_csv(data_path)
    
    # ดึงคอลัมน์ Features (ตัด Id และ Species ออก)
    feature_cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
    features = df[feature_cols].values
    
    # แปลงชื่อสายพันธุ์ให้เป็นตัวเลข 0, 1, 2
    classes = sorted(list(df['Species'].unique()))
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    labels = df['Species'].map(class_to_idx).values
    
    print("Detected classes:", classes)
    print(f"Loaded Iris dataset: {len(features)} samples")
    
    return features, labels, classes