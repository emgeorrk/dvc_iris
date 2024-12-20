import json
import pickle
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, auc, accuracy_score,
                             precision_recall_fscore_support, classification_report,
                             confusion_matrix)
from sklearn.preprocessing import label_binarize


def evaluate():
    # Load test data
    test_df = pd.read_csv('data/prepared/test.csv')
    X_test = test_df.drop('target', axis=1)
    y_test = test_df['target']

    # Load model
    with open('models/model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    class_report = classification_report(y_test, y_pred, output_dict=True)

    # Calculate per-class ROC AUC
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    roc_auc = {}
    for i in range(3):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
        roc_auc[f'class_{i}'] = auc(fpr, tpr)

    metrics = {
        'test_accuracy': accuracy,
        'test_precision': precision,
        'test_recall': recall,
        'test_f1': f1,
        'per_class_metrics': {
            f'class_{i}': {
                'precision': class_report[str(i)]['precision'],
                'recall': class_report[str(i)]['recall'],
                'f1-score': class_report[str(i)]['f1-score'],
                'support': class_report[str(i)]['support'],
                'roc_auc': roc_auc[f'class_{i}']
            } for i in range(3)
        }
    }

    # Save metrics
    with open('metrics/test_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)

    # Generate ROC curves
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red', 'green']
    for i in range(3):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
        plt.plot(fpr, tpr, color=colors[i],
                 label=f'ROC class {i} (AUC = {roc_auc[f"class_{i}"]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves per Class')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig('plots/roc_curve.png')
    plt.close()

    # Generate confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix (Test)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('plots/test_confusion_matrix.png')
    plt.close()


if __name__ == '__main__':
    evaluate()