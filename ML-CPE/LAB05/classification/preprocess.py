import numpy as np

def to_features(data):
    """แปลงประเภทข้อมูลให้อยู่ในรูปแบบ float32 พร้อมใช้วิเคราะห์"""
    return data.astype(np.float32)