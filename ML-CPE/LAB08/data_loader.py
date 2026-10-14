import numpy as np
from tensorflow.keras.datasets import fashion_mnist

def load_data(data_path=None, img_size=32, max_per_class=1000):
    print("Loading Fashion-MNIST dataset (Clothing & Shoes classes)...")
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    
    # แปลงภาพขาวดำ 1 แชนแนล เป็น 3 แชนแนล (RGB) เพื่อให้เข้ากับโมเดล VGG
    X_train = np.stack([X_train]*3, axis=-1)
    X_test = np.stack([X_test]*3, axis=-1)
    
    images = np.concatenate((X_train, X_test), axis=0)
    labels = np.concatenate((y_train, y_test), axis=0).flatten()
    
    # 10 คลาสของ Fashion-MNIST
    classes = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]
    
    # จำกัดจำนวนข้อมูลต่อคลาส (เช่น คลาสละ 1,000 รูป รวมเป็น 10,000 รูป) เพื่อความรวดเร็ว
    if max_per_class:
        filtered_images = []
        filtered_labels = []
        for c in range(len(classes)):
            indices = np.where(labels == c)[0][:max_per_class]
            filtered_images.append(images[indices])
            filtered_labels.append(labels[indices])
        images = np.concatenate(filtered_images, axis=0)
        labels = np.concatenate(filtered_labels, axis=0)
        
    print(f"Loaded {len(images)} images successfully across {len(classes)} classes.")
    return images, labels, classes