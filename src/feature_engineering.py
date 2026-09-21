import pandas as pd


SELECTED_FEATURES = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "GarageArea",
    "TotalBsmtSF",
    "1stFlrSF",
    "YearBuilt",
    "YearRemodAdd",
    "KitchenQual",
    "ExterQual",
    "BsmtQual",
    "GarageFinish",
    "2ndFlrSF",
    "FullBath",
    "TotRmsAbvGrd",
    "GarageYrBlt",
    "Fireplaces",
    "MasVnrArea",
    "BsmtFinSF1",
    "LotArea",
    "BsmtExposure",
    "GarageType",
    "HouseStyle",
    "Exterior1st",
    "SaleCondition",
]


ENGINEERED_FEATURES = [
    "TotalSF",
    "TotalBathrooms",
    "HouseAge",
    "YearsSinceRemod"
]


def load_data():
    """Load the cleaned training dataset."""
    return pd.read_csv("data/processed/train_cleaned.csv")


def create_features(df):
    """Create new features from existing variables."""

    df = df.copy()

    # Total surface area
    df["TotalSF"] = (
        df["TotalBsmtSF"]
        + df["1stFlrSF"]
        + df["2ndFlrSF"]
    )

    # Total number of bathrooms
    df["TotalBathrooms"] = (
        df["FullBath"]
        + 0.5 * df["HalfBath"]
        + df["BsmtFullBath"]
        + 0.5 * df["BsmtHalfBath"]
    )

    # Age of the house when it was sold
    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]

    # Years since the last renovation
    df["YearsSinceRemod"] = (
        df["YrSold"] - df["YearRemodAdd"]
    )

    # Prevent negative values
    df["YearsSinceRemod"] = df["YearsSinceRemod"].clip(lower=0)


    return df


def select_features(df):
    """Select original and engineered features."""

    selected_features = SELECTED_FEATURES + ENGINEERED_FEATURES

    X = df[selected_features].copy()
    y = df["SalePrice"].copy()

    return X, y



def check_engineered_features(X):
    """Check the coherence of engineered features."""

    print("\nEngineered features:")
    print(X[ENGINEERED_FEATURES].describe())

    print("\nNegative values:")

    for column in ENGINEERED_FEATURES:
        negative_count = (X[column] < 0).sum()

        print(f"{column}: {negative_count}")

def save_features(X, y):
    """Save the final feature-engineered dataset."""

    df_features = X.copy()
    df_features["SalePrice"] = y

    df_features.to_csv(
        "data/processed/train_features.csv",
        index=False,
    )

    print(
        "\nFeature-engineered data saved to:"
        " data/processed/train_features.csv"
    )

def main():
    df = load_data()

    # Create engineered features
    df = create_features(df)

    # Select final features
    X, y = select_features(df)

    print("Selected features:", len(X.columns))
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # Validate engineered features
    check_engineered_features(X)
    save_features(X, y)


if __name__ == "__main__":
    main()