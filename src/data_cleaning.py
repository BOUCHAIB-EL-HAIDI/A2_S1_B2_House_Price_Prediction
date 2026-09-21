import pandas as pd


def load_data():
    """Load the training dataset."""

    return pd.read_csv("data/raw/train.csv")


def handle_missing_values(df):
    """Handle missing values with a known meaning."""

    none_columns = [
        "PoolQC",
        "MiscFeature",
        "Alley",
        "Fence",
        "FireplaceQu",
        "GarageType",
        "GarageFinish",
        "GarageQual",
        "GarageCond",
        "BsmtQual",
        "BsmtCond",
        "BsmtExposure",
        "BsmtFinType1",
        "BsmtFinType2",
        "MasVnrType",
    ]

    df[none_columns] = df[none_columns].fillna("None")

    zero_columns = [
        "GarageArea",
        "GarageCars",
        "BsmtFinSF1",
        "BsmtFinSF2",
        "BsmtUnfSF",
        "TotalBsmtSF",
        "BsmtFullBath",
        "BsmtHalfBath",
        "MasVnrArea",
    ]

    df[zero_columns] = df[zero_columns].fillna(0)

    return df


def clean_data(df):
    """Apply all cleaning operations."""

    df = df.copy()

    df = handle_missing_values(df)

    return df
def save_cleaned_data(df):
    """Save the cleaned training dataset."""

    df.to_csv(
        "data/processed/train_cleaned.csv",
        index=False
    )


def main():

    df = load_data()

    print("Before cleaning:")
    print("Missing values:", df.isnull().sum().sum())

    df = clean_data(df)

    print("\nAfter cleaning:")
    print("Missing values:", df.isnull().sum().sum())
    save_cleaned_data(df)


if __name__ == "__main__":
    main()