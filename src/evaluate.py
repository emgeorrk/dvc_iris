import json
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


def evaluate():
    # Load test data
    test_df = pd.read_csv('data/prepared/test.csv')
    X_test = test_df.drop('target', axis=1)
    y_test = test_df['target']

    # Load trained model
    with open('models/model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Calculate metrics
    test_score = model.score(X_test, y_test)
    metrics = {
        'test_accuracy': test_score
    }

    # Save metrics
    with open('metrics/test_metrics.json', 'w') as f:
        json.dump(metrics, f)

    # Generate ROC curve
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    y_score = model.predict_proba(X_test)

    plt.figure(figsize=(8, 6))
    colors = ['blue', 'red', 'green']
    for i in range(3):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i],
                 label=f'ROC curve (class {i}) (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='lower right')
    plt.savefig('plots/roc_curve.png')
    plt.close()


if __name__ == '__main__':
    evaluate()