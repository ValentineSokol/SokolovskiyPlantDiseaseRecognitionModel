import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from .config import Config

class Evaluator:
    def __init__(self, model, config=None):
        self.model = model
        self.config = config or Config()
        
    def predict(self, X):
        predictions = self.model.predict(X)
        return predictions
    
    def predict_classes(self, X):
        predictions = self.predict(X)
        return np.argmax(predictions, axis=1)
    
    def evaluate(self, test_generator):
        results = self.model.evaluate(test_generator, verbose=1)
        loss, accuracy = results[0], results[1]
        
        print(f"\nTest Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")
        
        return {'loss': loss, 'accuracy': accuracy}
    
    def get_predictions(self, test_generator):
        test_generator.reset()
        y_pred = self.model.predict(test_generator, verbose=1)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = test_generator.classes
        
        return y_true, y_pred_classes, y_pred
    
    def calculate_metrics(self, y_true, y_pred_classes):
        cm = confusion_matrix(y_true, y_pred_classes)
        accuracy = accuracy_score(y_true, y_pred_classes)
        
        report = classification_report(
            y_true, 
            y_pred_classes,
            target_names=self.config.CLASSES,
            output_dict=True
        )
        
        return {
            'confusion_matrix': cm,
            'accuracy': accuracy,
            'classification_report': report
        }
    
    def print_metrics(self, metrics):
        print("\nConfusion Matrix:")
        print(metrics['confusion_matrix'])
        
        print(f"\nOverall Accuracy: {metrics['accuracy']:.4f}")
        
        print("\nClassification Report:")
        for class_name in self.config.CLASSES:
            if class_name in metrics['classification_report']:
                class_metrics = metrics['classification_report'][class_name]
                print(f"{class_name}:")
                print(f"  Precision: {class_metrics['precision']:.4f}")
                print(f"  Recall: {class_metrics['recall']:.4f}")
                print(f"  F1-Score: {class_metrics['f1-score']:.4f}")
    
    def analyze_errors(self, y_true, y_pred_classes, y_pred_proba):
        misclassified_idx = np.where(y_true != y_pred_classes)[0]
        
        print(f"\nTotal misclassified: {len(misclassified_idx)}")
        
        low_confidence = []
        for idx in misclassified_idx:
            confidence = np.max(y_pred_proba[idx])
            if confidence < 0.7:
                low_confidence.append(idx)
        
        print(f"Low confidence misclassifications: {len(low_confidence)}")
        
        return misclassified_idx, low_confidence
