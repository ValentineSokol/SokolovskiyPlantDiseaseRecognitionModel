import os
import numpy as np
from src import Config, DataLoader, Preprocessor, Augmentor, DiseaseClassifier, Trainer, Evaluator

def main():
    print("Initializing configuration...")
    config = Config()
    config.create_dirs()
    
    print("\nLoading data...")
    data_loader = DataLoader(config)
    images, labels = data_loader.load_from_directory(config.RAW_DATA_DIR)
    
    print(f"Loaded {len(images)} images")
    print(f"Classes distribution: {np.bincount(labels)}")
    
    print("\nPreprocessing data...")
    preprocessor = Preprocessor(config)
    images = preprocessor.preprocess_batch(images, normalize=True, enhance=False, denoise=False)
    
    print("\nSplitting data...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = data_loader.split_data(images, labels)
    
    print(f"Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
    
    print("\nCalculating class weights...")
    class_weights = data_loader.get_class_weights(y_train)
    print(f"Class weights: {class_weights}")
    
    print("\nCreating data generators...")
    augmentor = Augmentor(config)
    train_gen = augmentor.get_train_generator(X_train, y_train)
    val_gen = augmentor.get_validation_generator(X_val, y_val)
    test_gen = augmentor.get_test_generator(X_test, y_test)
    
    print("\nBuilding model...")
    classifier = DiseaseClassifier(config)
    model = classifier.get_model()
    classifier.summary()
    
    print("\nTraining model...")
    trainer = Trainer(model, config)
    history = trainer.train(train_gen, val_gen, class_weights=class_weights)
    
    history_path = os.path.join(config.MODELS_DIR, 'training_history.pkl')
    trainer.save_history(history_path)
    print(f"Training history saved to {history_path}")
    
    print("\nEvaluating model...")
    evaluator = Evaluator(model, config)
    
    test_results = evaluator.evaluate(test_gen)
    
    y_true, y_pred_classes, y_pred_proba = evaluator.get_predictions(test_gen)
    
    metrics = evaluator.calculate_metrics(y_true, y_pred_classes)
    evaluator.print_metrics(metrics)
    
    misclassified, low_conf = evaluator.analyze_errors(y_true, y_pred_classes, y_pred_proba)
    
    final_model_path = os.path.join(config.MODELS_DIR, 'final_model.h5')
    classifier.save_model(final_model_path)
    print(f"\nFinal model saved to {final_model_path}")
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()
