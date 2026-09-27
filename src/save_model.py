import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline

from src.preprocessing import create_preprocessor


def load_data():
    return pd.read_csv(
        "data/processed/train_features_no_outliers.csv"
    )


def prepare_data(df):
    X = df.drop(columns=["SalePrice"])
    y = df["SalePrice"]

    return X, y


def create_model(X):
    preprocessor = create_preprocessor(X)

    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
    )

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model),
    ])

    return pipeline


def main():
    df = load_data()

    X, y = prepare_data(df)

    pipeline = create_model(X)

    pipeline.fit(X, y)

    joblib.dump(
        pipeline,
        "models/gradient_boosting_final.joblib"
    )

    print(
        "Model saved to:",
        "models/gradient_boosting_final.joblib"
    )


if __name__ == "__main__":
    main()