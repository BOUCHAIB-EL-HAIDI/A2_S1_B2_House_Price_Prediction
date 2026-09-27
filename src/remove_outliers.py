import pandas as pd


def load_data():
    return pd.read_csv(
        "data/processed/train_features.csv"
    )


def remove_outliers(df):
    outlier_features = [
        "GrLivArea",
        "LotArea",
        "TotalBsmtSF",
        "1stFlrSF",
        "GarageArea",
    ]

    mask = pd.Series(True, index=df.index)

    for column in outlier_features:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)

        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        mask &= (
            (df[column] >= lower_bound)
            & (df[column] <= upper_bound)
        )

        print(f"\n{column}")
        print(f"Q1: {q1}")
        print(f"Q3: {q3}")
        print(f"IQR: {iqr}")
        print(f"Lower bound: {lower_bound}")
        print(f"Upper bound: {upper_bound}")

    return df[mask].copy()


def main():
    df = load_data()

    print("Original shape:", df.shape)

    df_without_outliers = remove_outliers(df)

    print(
        "\nShape after removing outliers:",
        df_without_outliers.shape
    )

    print(
        "Removed rows:",
        len(df) - len(df_without_outliers)
    )

    output_path = (
        "data/processed/train_features_no_outliers.csv"
    )

    df_without_outliers.to_csv(
        output_path,
        index=False
    )

    print(
        "\nSaved to:",
        output_path
    )


if __name__ == "__main__":
    main() 