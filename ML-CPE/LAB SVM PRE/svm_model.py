from sklearn.svm import SVC

def train_svm_model(X_train, y_train):
    # สร้างโมเดล AI ประเภท Support Vector Machine (SVM) โดยใช้เส้นแบ่งแบบ Linear (เชิงเส้น)
    model = SVC(kernel='linear')
    
    # สั่งให้โมเดลเรียนรู้ (Fit) จากข้อมูลฝึกสอน (X_train และ y_train)
    model.fit(X_train, y_train)
    
    return model