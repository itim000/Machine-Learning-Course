from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_data(df):
    # สร้างตัวแปลงข้อความให้เป็นตัวเลขด้วยวิธี TF-IDF และตัดคำฟุ่มเฟือยภาษาอังกฤษออก (stop_words)
    vectorizer = TfidfVectorizer(stop_words='english')
    
    # แปลงข้อความทั้งหมดในคอลัมน์ text ให้กลายเป็นตารางเวกเตอร์ตัวเลข (เก็บไว้ในตัวแปร X)
    X = vectorizer.fit_transform(df['text'])
    
    # ดึงป้ายกำกับประเภท (spam หรือ ham) มาเก็บไว้เป็นตัวแปรเฉลย (y)
    y = df['label']
    
    return X, y, vectorizer