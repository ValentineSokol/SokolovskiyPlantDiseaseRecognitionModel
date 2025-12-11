import tensorflow as tf
from tensorflow.keras import layers, models
from .config import Config

class DiseaseClassifier:
    def __init__(self, config=None):
        self.config = config or Config()
        self.model = None
        
    def build_model(self):
        inputs = layers.Input(shape=self.config.INPUT_SHAPE)
        x = inputs
        
        filters = self.config.INITIAL_FILTERS
        for i in range(self.config.CONV_BLOCKS):
            x = layers.Conv2D(filters, (3, 3), padding='same')(x)
            if self.config.USE_BATCH_NORM:
                x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.MaxPooling2D((2, 2))(x)
            if self.config.USE_DROPOUT and i > 0:
                x = layers.Dropout(0.25)(x)
            filters *= 2
        
        x = layers.Flatten()(x)
        x = layers.Dense(self.config.FC_UNITS)(x)
        if self.config.USE_BATCH_NORM:
            x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        if self.config.USE_DROPOUT:
            x = layers.Dropout(self.config.DROPOUT_RATE)(x)
        
        outputs = layers.Dense(self.config.NUM_CLASSES, activation='softmax')(x)
        
        self.model = models.Model(inputs=inputs, outputs=outputs)
        return self.model
    
    def compile_model(self, learning_rate=None):
        if self.model is None:
            self.build_model()
            
        lr = learning_rate or self.config.LEARNING_RATE
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
    def get_model(self):
        if self.model is None:
            self.build_model()
            self.compile_model()
        return self.model
    
    def summary(self):
        if self.model is None:
            self.build_model()
        return self.model.summary()
    
    def save_model(self, filepath):
        if self.model is not None:
            self.model.save(filepath)
    
    def load_model(self, filepath):
        self.model = models.load_model(filepath)
        return self.model
