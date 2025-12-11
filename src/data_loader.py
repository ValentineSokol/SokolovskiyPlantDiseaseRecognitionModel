import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from .config import Config

class DataLoader:
    def __init__(self, config=None):
        self.config = config or Config()
        
    def load_from_directory(self, directory):
        images = []
        labels = []
        
        for class_idx, class_name in enumerate(self.config.CLASSES):
            class_dir = os.path.join(directory, class_name)
            if not os.path.exists(class_dir):
                continue
                
            for img_name in os.listdir(class_dir):
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                    
                img_path = os.path.join(class_dir, img_name)
                try:
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize((self.config.IMG_WIDTH, self.config.IMG_HEIGHT))
                    images.append(np.array(img))
                    labels.append(class_idx)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    
        return np.array(images), np.array(labels)
    
    def split_data(self, images, labels):
        X_temp, X_test, y_temp, y_test = train_test_split(
            images, labels, 
            test_size=self.config.TEST_SPLIT,
            stratify=labels,
            random_state=42
        )
        
        val_size = self.config.VALIDATION_SPLIT / (1 - self.config.TEST_SPLIT)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size,
            stratify=y_temp,
            random_state=42
        )
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def get_class_weights(self, labels):
        unique, counts = np.unique(labels, return_counts=True)
        total = len(labels)
        weights = {i: total / (len(unique) * count) for i, count in zip(unique, counts)}
        return weights
