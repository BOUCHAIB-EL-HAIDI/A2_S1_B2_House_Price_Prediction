
import pandas as pd


pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)


def load_data():
    """Load the housing dataset."""

    df = pd.read_csv("data/raw/train.csv")

    return df


def show_overview(df):
    """Display the general structure of the dataset."""

    print("Dataset loaded successfully.")
    print("Shape:", df.shape)

    print("\nNumber of rows:", df.shape[0])
    print("Number of columns:", df.shape[1])

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())


def analyze_columns(df):
    """Identify numerical and categorical columns."""

    numeric_columns = df.select_dtypes(
        include=["int64", "float64"]
    ).columns

    categorical_columns = df.select_dtypes(
        include=["str", "object"]
    ).columns

    print("\nNumber of numerical columns:", len(numeric_columns))

    print("\nNumerical columns:")
    print(numeric_columns.tolist())

    print("\nNumber of categorical columns:", len(categorical_columns))

    print("\nCategorical columns:")
    print(categorical_columns.tolist())

    return numeric_columns, categorical_columns


def analyze_missing_values(df):
    """Analyze missing values and their percentages."""

    missing_summary = pd.DataFrame({
        "missing_count": df.isnull().sum(),
        "missing_percent": df.isnull().mean() * 100
    })

    missing_summary = missing_summary[
        missing_summary["missing_count"] > 0
    ].sort_values(
        "missing_count",
        ascending=False
    )

    print("\nMissing values:")
    print(missing_summary)


def check_duplicates(df):
    """Check for duplicated rows."""

    print("\nNumber of duplicated rows:")
    print(df.duplicated().sum())


def analyze_numeric_statistics(df, numeric_columns):
    """Display descriptive statistics for numerical features."""

    print("\nNumerical statistics:")
    print(df[numeric_columns].describe())


def analyze_saleprice(df):
    """Analyze the SalePrice target variable."""

    print("\nSalePrice statistics:")
    print(df["SalePrice"].describe())

    print("\nMinimum SalePrice:", df["SalePrice"].min())
    print("Maximum SalePrice:", df["SalePrice"].max())
    print("Mean SalePrice:", df["SalePrice"].mean())
    print("Median SalePrice:", df["SalePrice"].median())

    print("\nSalePrice <= 0:")
    print((df["SalePrice"] <= 0).sum())

    print("\nMissing SalePrice:")
    print(df["SalePrice"].isnull().sum())


def check_consistency(df):
    """Check for inconsistent or invalid numerical values."""

    checks = {
        "LotArea < 0": (df["LotArea"] < 0).sum(),
        "GrLivArea < 0": (df["GrLivArea"] < 0).sum(),
        "TotalBsmtSF < 0": (df["TotalBsmtSF"] < 0).sum(),
        "GarageArea < 0": (df["GarageArea"] < 0).sum(),
        "GarageCars < 0": (df["GarageCars"] < 0).sum(),
        "OverallQual outside 1-10": (
            (df["OverallQual"] < 1)
            | (df["OverallQual"] > 10)
        ).sum(),
    }

    print("\nConsistency checks:")

    for check, count in checks.items():
        print(f"{check}: {count}")


def analyze_saleprice_outliers(df):
    """Identify potential SalePrice outliers using the IQR method."""

    q1 = df["SalePrice"].quantile(0.25)
    q3 = df["SalePrice"].quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    saleprice_outliers = df[
        (df["SalePrice"] < lower_bound)
        | (df["SalePrice"] > upper_bound)
    ]

    print("\nSalePrice outliers:")
    print("Q1:", q1)
    print("Q3:", q3)
    print("IQR:", iqr)
    print("Lower bound:", lower_bound)
    print("Upper bound:", upper_bound)
    print("Potential outliers:", len(saleprice_outliers))

    print("\nLow-price outliers:")

    print(
        saleprice_outliers[
            saleprice_outliers["SalePrice"] < lower_bound
        ][
            [
                "Id",
                "OverallQual",
                "GrLivArea",
                "YearBuilt",
                "SalePrice",
            ]
        ]
        .sort_values("SalePrice")
        .head(10)
    )

    print("\nHigh-price outliers:")

    print(
        saleprice_outliers[
            saleprice_outliers["SalePrice"] > upper_bound
        ][
            [
                "Id",
                "OverallQual",
                "GrLivArea",
                "YearBuilt",
                "SalePrice",
            ]
        ]
        .sort_values("SalePrice", ascending=False)
        .head(10)
    )


def analyze_feature_outliers(df):
    """Identify potential outliers in selected numerical features."""

    features_to_check = [
        "GrLivArea",
        "LotArea",
        "TotalBsmtSF",
        "1stFlrSF",
        "GarageArea",
    ]

    print("\nFeature outliers:")

    for column in features_to_check:

        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)

        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = df[
            (df[column] < lower_bound)
            | (df[column] > upper_bound)
        ]

        print(
            f"{column}: "
            f"{len(outliers)} potential outliers"
        )


def show_extreme_values(df):
    """Display houses with the largest living areas."""

    print("\nExtreme GrLivArea values:")

    print(
        df[
            [
                "Id",
                "GrLivArea",
                "OverallQual",
                "YearBuilt",
                "SalePrice",
            ]
        ]
        .sort_values("GrLivArea", ascending=False)
        .head(10)
    )


def show_summary(df):
    """Display a final exploration summary."""

    print("\nExploration completed.")

    print("Shape:", df.shape)
    print("Duplicates:", df.duplicated().sum())
    print("Missing values:", df.isnull().sum().sum())
    print("Minimum SalePrice:", df["SalePrice"].min())
    print("Maximum SalePrice:", df["SalePrice"].max())
    print("Mean SalePrice:", df["SalePrice"].mean())
    print("Median SalePrice:", df["SalePrice"].median())


def main():

    df = load_data()

    show_overview(df)

    numeric_columns, categorical_columns = analyze_columns(df)

    analyze_missing_values(df)

    check_duplicates(df)

    analyze_numeric_statistics(
        df,
        numeric_columns
    )

    analyze_saleprice(df)

    check_consistency(df)

    analyze_saleprice_outliers(df)

    analyze_feature_outliers(df)

    show_extreme_values(df)

    show_summary(df)
    print(df["SalePrice"].head())
    print(df["SalePrice"].tail())


if __name__ == "__main__":
    main()

