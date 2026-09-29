from sklearn.model_selection import train_test_split

def split_dataset(X, y, test_size=0.2, val_size=0.1):
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    adjusted_val_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=adjusted_val_size, random_state=42, stratify=y_train_val
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test