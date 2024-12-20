import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris


def prepare_data():
    # Load Iris dataset
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target

    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Save splits
    train_df.to_csv('data/prepared/train.csv', index=False)
    test_df.to_csv('data/prepared/test.csv', index=False)


if __name__ == '__main__':
    prepare_data()