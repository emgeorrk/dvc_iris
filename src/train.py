import json
import pickle
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

def train_model():
    # Load parameters from params.yaml
    with open('params.yaml') as f:
        params = yaml.safe_load(f)

    # Load training data
    train_df = pd.read_csv('data/prepared/train.csv')
    X = train_df.drop('target', axis=1)
    y = train_df['target']

    # Modify the model to make it less accurate
    model = RandomForestClassifier(
        n_estimators=5,  # Reduced number of trees to make model less powerful
        max_depth=2,     # Limit the depth of the trees to make them less complex
        random_state=42
    )
    model.fit(X, y)

    # Save model
    with open('models/model.pkl', 'wb') as f:
        pickle.dump(model, f)

    # Calculate and save metrics
    train_pred = model.predict(X)
    train_score = model.score(X, y)
    metrics = {
        'train_accuracy': train_score
    }
    with open('metrics/train_metrics.json', 'w') as f:
        json.dump(metrics, f)

    # Generate plots
    # Confusion Matrix
    cm = confusion_matrix(y, train_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('Confusion Matrix')
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
