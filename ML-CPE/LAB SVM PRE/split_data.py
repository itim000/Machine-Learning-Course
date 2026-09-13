from sklearn.model_selection import train_test_split

def split_dataset(X, y):
    # แบ่งข้อมูลเป็น 2 ส่วน: ส่วนฝึกสอน (Train 80%) และส่วนทดสอบ (Test 20%) 
    # กำหนด random_state=42 เพื่อให้คอมพิวเตอร์สุ่มแบ่งข้อมูลชุดเดิมทุกครั้งที่รัน
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test