import pandas as pd

def load_heart_data(filepath='heart.csv'):
    """โหลดข้อมูลโรคหัวใจจากไฟล์ CSV"""
    df = pd.read_csv(filepath)
    
    # แยก Features (X) และ Target (y)
    X = df.drop(columns=['target']).values
    y = df['target'].values
    
    return X, y