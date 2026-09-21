import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


def load_data():
    """Load the feature-engineered dataset."""

    return pd.read_csv(
        "data/processed/train_features.csv"
    )


def prepare_data(df):
    """Separate features from the target."""

    X = df.drop(columns=["SalePrice"])
    y = df["SalePrice"]

    return X, y


def create_preprocessor(X):
    """Create the preprocessing pipeline."""

    numeric_columns = X.select_dtypes(
        include=["int64", "float64"]
    ).columns

    categorical_columns = X.select_dtypes(
        include=["object", "str"]
    ).columns

    print("Numerical columns:", len(numeric_columns))
    print("Categorical columns:", len(categorical_columns))

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        (
            "numerical",
            numeric_pipeline,
            numeric_columns,
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_columns,
        ),
    ])

    return preprocessor


def main():
    df = load_data()

    X, y = prepare_data(df)

    preprocessor = create_preprocessor(X)

    X_processed = preprocessor.fit_transform(X)

    print("\nOriginal X shape:", X.shape)
    print("Processed X shape:", X_processed.shape)


if __name__ == "__main__":
    main()