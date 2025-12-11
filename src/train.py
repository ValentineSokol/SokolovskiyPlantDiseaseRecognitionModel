import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from .config import Config

class Trainer:
    def __init__(self, model, config=None):
        self.model = model
        self.config = config or Config()
        self.history = None
        
    def create_callbacks(self, model_path):
        callbacks = []
        
        checkpoint = ModelCheckpoint(
            filepath=model_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        callbacks.append(checkpoint)
        
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stop)
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        return callbacks
    
    def train(self, train_generator, validation_generator, epochs=None, class_weights=None):
        epochs = epochs or self.config.EPOCHS
        
        model_path = os.path.join(self.config.MODELS_DIR, 'best_model.h5')
        callbacks = self.create_callbacks(model_path)
        
        steps_per_epoch = len(train_generator)
        validation_steps = len(validation_generator)
        
        self.history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=callbacks,
            class_weight=class_weights,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            verbose=1
        )
        
        return self.history
    
    def get_history(self):
        return self.history
    
    def save_history(self, filepath):
        if self.history is not None:
            import pickle
            with open(filepath, 'wb') as f:
                pickle.dump(self.history.history, f)
    
    def load_history(self, filepath):
        import pickle
        with open(filepath, 'rb') as f:
            self.history = pickle.load(f)
        return self.history
