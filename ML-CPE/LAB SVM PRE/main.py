# นำเข้าฟังก์ชันการทำงานจากไฟล์ย่อยทั้ง 6 ไฟล์ที่เราสร้างไว้
from data_load import load_data
from preprocess import preprocess_data
from split_data import split_dataset
from svm_model import train_svm_model
from evaluate import evaluate_model
from test_svm import test_new_messages

def main():
    print("1. กำลังโหลดข้อมูลจากไฟล์ในเครื่อง...")
    df = load_data()
    
    print("2. กำลังแปลงข้อความด้วย TF-IDF...")
    X, y, vectorizer = preprocess_data(df)
    
    print("3. กำลังแบ่งข้อมูล Train / Test...")
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    
    print("4. กำลังฝึกสอนโมเดล SVM...")
    model = train_svm_model(X_train, y_train)
    
    print("\n5. กำลังประเมินผลโมเดลและเซฟรูป...")
    evaluate_model(model, X_test, y_test)
    
    print("\n6. กำลังทดสอบข้อความใหม่และเซฟรูป...")
    test_new_messages(model, vectorizer)
    
    print("\nเสร็จสิ้นทุกกระบวนการ!")

if __name__ == "__main__":
    main()