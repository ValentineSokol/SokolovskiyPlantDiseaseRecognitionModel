import numpy as np
import cv2
from .config import Config

class Preprocessor:
    def __init__(self, config=None):
        self.config = config or Config()
        
    def normalize(self, images):
        return images.astype('float32') / 255.0
    
    def denormalize(self, images):
        return (images * 255.0).astype('uint8')
    
    def resize_batch(self, images, size=None):
        if size is None:
            size = (self.config.IMG_WIDTH, self.config.IMG_HEIGHT)
        
        resized = []
        for img in images:
            resized.append(cv2.resize(img, size))
        return np.array(resized)
    
    def enhance_contrast(self, images):
        enhanced = []
        for img in images:
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            enhanced_lab = cv2.merge([l, a, b])
            enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
            enhanced.append(enhanced_rgb)
        return np.array(enhanced)
    
    def remove_noise(self, images, kernel_size=5):
        denoised = []
        for img in images:
            blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
            denoised.append(blurred)
        return np.array(denoised)
    
    def preprocess_batch(self, images, normalize=True, enhance=False, denoise=False):
        processed = images.copy()
        
        if denoise:
            processed = self.remove_noise(processed)
        if enhance:
            processed = self.enhance_contrast(processed)
        if normalize:
            processed = self.normalize(processed)
            
        return processed
