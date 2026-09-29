import os
import cv2
import numpy as np

def load_data(data_path, img_size=48, max_per_class=None):
    images = []
    labels = []
    classes = []
    
    if not os.path.exists(data_path):
        return images, labels, classes

    subdirs = [os.path.join(data_path, d) for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    
    target_dirs = []
    for d in subdirs:
        if os.path.basename(d).lower() in ['train', 'test', 'val']:
            inner_dirs = [os.path.join(d, inner) for inner in os.listdir(d) if os.path.isdir(os.path.join(d, inner))]
            target_dirs.extend(inner_dirs)
        else:
            target_dirs.append(d)

    class_names = sorted(list(set([os.path.basename(td) for td in target_dirs if os.listdir(td)])))
    classes = class_names
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

    print(f"พบหมวดหมู่ (Classes): {classes}")

    for td in target_dirs:
        cls_name = os.path.basename(td)
        if cls_name not in class_to_idx:
            continue
        
        class_idx = class_to_idx[cls_name]
        files = os.listdir(td)
        
        if max_per_class:
            files = files[:max_per_class]

        count = 0
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                file_path = os.path.join(td, file)
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (img_size, img_size))
                    images.append(img)
                    labels.append(class_idx)
                    count += 1
        
        if count > 0:
            print(f" - โหลดคลาส '{cls_name}' สำเร็จจำนวน {count} รูปภาพ")

    return images, labels, classes