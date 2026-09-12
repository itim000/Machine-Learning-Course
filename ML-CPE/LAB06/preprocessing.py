from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def prepare_data(X, y):
    """แบ่งข้อมูล Train/Val/Test และปรับ Standard Scaling"""
    # 1. แบ่ง Train 70%, Temp 30%
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    # 2. แบ่ง Temp เป็น Validation 15% และ Test 15%
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    # 3. Standardize ข้อมูลตัวเลข
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    return X_train, y_train, X_val, y_val, X_test, y_test