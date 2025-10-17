import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve
)


def calculate_metrics(y_true, y_pred, y_proba, dataset_name=""):
    """
    * Calculate and print evaluation metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities
        dataset_name: Name of dataset for logging
    
    Returns:
        Dictionary containing all metrics
    """
    acc = accuracy_score(y_true, y_pred) * 100
    prec = precision_score(y_true, y_pred, average='binary') * 100
    rec = recall_score(y_true, y_pred, average='binary') * 100
    f1 = f1_score(y_true, y_pred, average='binary') * 100
    auc_score = roc_auc_score(y_true, y_proba[:, 1]) * 100
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"\n{dataset_name} Results:")
    print(f"  Accuracy:  {acc:.2f}%")
    print(f"  Precision: {prec:.2f}%")
    print(f"  Recall:    {rec:.2f}%")
    print(f"  F1-Score:  {f1:.2f}%")
    print(f"  AUC:       {auc_score:.2f}%")
    
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'auc': auc_score,
        'confusion_matrix': cm,
        'predictions': y_pred,
        'proba': y_proba
    }


def evaluate_model(classifier, scaler, test_features, test_labels, dataset_name="Test"):
    """
    * Evaluate classifier on test data
    
    Args:
        classifier: Trained classifier
        scaler: Fitted StandardScaler
        test_features: Test features
        test_labels: Test labels
        dataset_name: Name for logging
    
    Returns:
        Dictionary containing evaluation metrics
    """

    test_features_scaled = scaler.transform(test_features)    
    test_pred = classifier.predict(test_features_scaled)
    test_proba = classifier.predict_proba(test_features_scaled)
    
    results = calculate_metrics(test_labels, test_pred, test_proba, dataset_name)
    
    return results


def print_classification_report(y_true, y_pred, labels=['Pneumonia (0)', 'Normal (1)']):
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=labels))


def get_roc_data(y_true, y_proba):
    fpr, tpr, thresholds = roc_curve(y_true, y_proba[:, 1])
    return fpr, tpr, thresholds
