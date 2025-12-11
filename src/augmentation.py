import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from .config import Config

class Augmentor:
    def __init__(self, config=None):
        self.config = config or Config()
        
    def create_generator(self, use_augmentation=True):
        if use_augmentation:
            return ImageDataGenerator(
                rotation_range=self.config.ROTATION_RANGE,
                width_shift_range=self.config.WIDTH_SHIFT_RANGE,
                height_shift_range=self.config.HEIGHT_SHIFT_RANGE,
                horizontal_flip=self.config.HORIZONTAL_FLIP,
                vertical_flip=self.config.VERTICAL_FLIP,
                zoom_range=self.config.ZOOM_RANGE,
                fill_mode='nearest'
            )
        else:
            return ImageDataGenerator()
    
    def get_train_generator(self, X_train, y_train):
        datagen = self.create_generator(use_augmentation=self.config.USE_AUGMENTATION)
        return datagen.flow(
            X_train, y_train,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True
        )
    
    def get_validation_generator(self, X_val, y_val):
        datagen = self.create_generator(use_augmentation=False)
        return datagen.flow(
            X_val, y_val,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False
        )
    
    def get_test_generator(self, X_test, y_test):
        datagen = self.create_generator(use_augmentation=False)
        return datagen.flow(
            X_test, y_test,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False
        )
