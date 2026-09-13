import pandas as pd

def load_data():
    # โหลดไฟล์ CSV จากในเครื่อง (ใช้ encoding='latin-1' เพื่อป้องกันตัวอักษรภาษาอังกฤษเพี้ยน)
    df = pd.read_csv("spam.csv", encoding='latin-1')
    
    # เลือกเฉพาะคอลัมน์ v1 (ประเภท: spam/ham) และ v2 (เนื้อหาข้อความ) ที่ใช้งานจริง
    df = df[['v1', 'v2']]
    
    # เปลี่ยนชื่อคอลัมน์เพื่อให้เข้าใจง่ายขึ้นเป็น label และ text
    df.columns = ['label', 'text']
    
    return df