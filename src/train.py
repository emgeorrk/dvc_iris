import json
import pickle
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support


def train_model():
    # Load parameters
    with open('params.yaml') as f:
        params = yaml.safe_load(f)

    # Load training data
    train_df = pd.read_csv('data/prepared/train.csv')
    X = train_df.drop('target', axis=1)
    y = train_df['target']

    # Train model
    model = RandomForestClassifier(
        n_estimators=params['train']['n_estimators'],
        max_depth=params['train']['max_depth'],
        random_state=42
    )
    model.fit(X, y)

    # Save model
    with open('models/model.pkl', 'wb') as f:
        pickle.dump(model, f)

    # Calculate metrics
    train_pred = model.predict(X)
    accuracy = accuracy_score(y, train_pred)
    precision, recall, f1, support = precision_recall_fscore_support(y, train_pred, average='weighted')

    # Get per-class metrics
    class_report = classification_report(y, train_pred, output_dict=True)

    metrics = {
        'train_accuracy': accuracy,
        'train_precision': precision,
        'train_recall': recall,
        'train_f1': f1,
        'per_class_metrics': {
            f'class_{i}': {
                'precision': class_report[str(i)]['precision'],
                'recall': class_report[str(i)]['recall'],
                'f1-score': class_report[str(i)]['f1-score'],
                'support': class_report[str(i)]['support']
            } for i in range(3)
        },
        'model_params': {
            'n_estimators': params['train']['n_estimators'],
            'max_depth': params['train']['max_depth']
        }
    }

    with open('metrics/train_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)

    # Generate plots
    # Confusion Matrix
    cm = confusion_matrix(y, train_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix (Training)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('plots/confusion_matrix.png')
    plt.close()

    # Feature Importance
    plt.figure(figsize=(10, 6))
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    sns.barplot(data=importance, x='importance', y='feature')
    plt.title('Feature Importance')
    plt.savefig('plots/feature_importance.png')
    plt.close()


if __name__ == '__main__':
    train_model()