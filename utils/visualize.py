"""
Visualization utilities for CAX-Net
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve
from config import Config


def plot_confusion_matrices(train_results, test_results, 
                           labels=Config.LABELS, save_path='results'):
    """Plot confusion matrices for train and test sets"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.heatmap(train_results['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=axes[0])
    axes[0].set_title(f'Training Confusion Matrix\nAccuracy: {train_results["accuracy"]:.2f}%')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    
    sns.heatmap(test_results['confusion_matrix'], annot=True, fmt='d', cmap='Greens',
                xticklabels=labels, yticklabels=labels, ax=axes[1])
    axes[1].set_title(f'Test Confusion Matrix\nAccuracy: {test_results["accuracy"]:.2f}%')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
    plt.show()


def plot_performance_comparison(train_results, test_results, save_path='results'):
    """Plot performance metrics comparison"""
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
    train_vals = [train_results['accuracy'], train_results['precision'],
                  train_results['recall'], train_results['f1_score'], train_results['auc']]
    test_vals = [test_results['accuracy'], test_results['precision'],
                 test_results['recall'], test_results['f1_score'], test_results['auc']]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, train_vals, width, label='Train', color='skyblue')
    bars2 = ax.bar(x + width/2, test_vals, width, label='Test', color='lightcoral')
    
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('CAX-Net Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    ax.set_ylim([0, 105])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()


def plot_roc_curve(train_results, test_results, train_labels, test_labels, save_path='results'):
    """Plot ROC curves"""
    fpr_train, tpr_train, _ = roc_curve(train_labels, train_results['proba'][:, 1])
    fpr_test, tpr_test, _ = roc_curve(test_labels, test_results['proba'][:, 1])
    
    plt.figure(figsize=(10, 6))
    plt.plot(fpr_train, tpr_train, linewidth=2, 
             label=f'Train (AUC = {train_results["auc"]:.2f}%)', color='blue')
    plt.plot(fpr_test, tpr_test, linewidth=2, 
             label=f'Test (AUC = {test_results["auc"]:.2f}%)', color='red')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - CAX-Net', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, 'roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.show()


def plot_feature_importance(classifier, latent_dim=Config.LATENT_DIM, save_path='results'):
    """Plot feature importance from XGBoost"""
    importance = classifier.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    plt.figure(figsize=(10, 5))
    plt.bar(range(latent_dim), importance[indices], color='teal', alpha=0.7)
    plt.xlabel('Encoded Feature Index', fontsize=12)
    plt.ylabel('Importance Score', fontsize=12)
    plt.title('XGBoost Feature Importance (10-Dimensional Latent Space)', 
              fontsize=12, fontweight='bold')
    plt.xticks(range(latent_dim), indices)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, 'feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.show()


def plot_sample_predictions(x_test, test_labels, test_pred, 
                           labels=Config.LABELS, save_path='results'):
    """Plot sample predictions"""
    correct_indices = np.where(test_pred == test_labels)[0]
    incorrect_indices = np.where(test_pred != test_labels)[0]
    
    fig, axes = plt.subplots(2, 6, figsize=(18, 6))
    fig.suptitle('CAX-Net Predictions on Test Set', fontsize=16, fontweight='bold')
    
    # Correct predictions
    for i, idx in enumerate(correct_indices[:6]):
        axes[0, i].imshow(x_test[idx], cmap='gray')
        axes[0, i].set_title(f'Pred: {labels[test_pred[idx]]}\nTrue: {labels[test_labels[idx]]}\n✓',
                            fontsize=10, color='green')
        axes[0, i].axis('off')
    
    # Incorrect predictions
    for i, idx in enumerate(incorrect_indices[:6] if len(incorrect_indices) >= 6 else incorrect_indices):
        axes[1, i].imshow(x_test[idx], cmap='gray')
        axes[1, i].set_title(f'Pred: {labels[test_pred[idx]]}\nTrue: {labels[test_labels[idx]]}\n✗',
                            fontsize=10, color='red')
        axes[1, i].axis('off')
    
    # Hide extra subplots if not enough incorrect predictions
    for i in range(len(incorrect_indices), 6):
        axes[1, i].axis('off')
    
    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, 'sample_predictions.png'), dpi=300, bbox_inches='tight')
    plt.show()