import os

class Config:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
    MODELS_DIR = os.path.join(BASE_DIR, 'models', 'saved')
    
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    IMG_CHANNELS = 3
    INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    
    CLASSES = ['healthy', 'angular_leaf_spot', 'rust', 'other_diseases']
    NUM_CLASSES = len(CLASSES)
    
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    VALIDATION_SPLIT = 0.2
    TEST_SPLIT = 0.1
    
    CONV_BLOCKS = 5
    INITIAL_FILTERS = 32
    DROPOUT_RATE = 0.5
    FC_UNITS = 256
    
    USE_BATCH_NORM = True
    USE_DROPOUT = True
    USE_AUGMENTATION = True
    
    ROTATION_RANGE = 20
    WIDTH_SHIFT_RANGE = 0.2
    HEIGHT_SHIFT_RANGE = 0.2
    HORIZONTAL_FLIP = True
    VERTICAL_FLIP = True
    ZOOM_RANGE = 0.2
    
    @classmethod
    def create_dirs(cls):
        for directory in [cls.RAW_DATA_DIR, cls.PROCESSED_DATA_DIR, cls.MODELS_DIR]:
            os.makedirs(directory, exist_ok=True)
