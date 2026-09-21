import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_data():
     return pd.read_csv("data/raw/train.csv")


def plot_price_distribution(df):
    plt.figure(figsize=(10, 6))

    sns.histplot(
        data=df,
        x="SalePrice",
        kde=True
    )

    plt.title("Distribution of House Prices")
    plt.xlabel("Sale Price")
    plt.ylabel("Number of Houses")

    plt.show()

# 1. SalePrice distribution
# Most houses are priced between Q1 = 129,975 and Q3 = 214,000.
# Mean = 180,921, median = 163,000, maximum = 755,000.
# The distribution is right-skewed with high-price observations.



def surface_vs_price(df):
    plt.figure(figsize=(10, 6))

    sns.scatterplot(
    data=df,
    x="GrLivArea",
    y="SalePrice"
    )
   
    plt.title("surface area vs saling price")
    plt.xlabel("surface area")
    plt.ylabel("selling price")
    plt.show()

# 2. GrLivArea vs SalePrice
# GrLivArea ranges from about 334 to 5,642 sq ft.
# There is a positive relationship between living area and SalePrice,
# but some very large houses have relatively low prices.


    
def plot_quality_vs_price(df):
    plt.figure(figsize=(10, 6))

    sns.boxplot(
        data=df,
        x="OverallQual",
        y="SalePrice"
    )

    plt.title("Overall Quality vs Sale Price")
    plt.xlabel("Overall Quality")
    plt.ylabel("Sale Price")

    plt.show()


# 3. OverallQual vs SalePrice
# OverallQual ranges from 1 to 10.
# Higher quality levels generally have higher SalePrice values.
# This suggests that OverallQual is an important feature for prediction.


def plot_neighborhood_vs_price(df):
    plt.figure(figsize=(14, 6))

    sns.boxplot(
        data=df,
        x="Neighborhood",
        y="SalePrice"
    )

    plt.title("Neighborhood vs Sale Price")
    plt.xlabel("Neighborhood")
    plt.ylabel("Sale Price")

    plt.xticks(rotation=45)

    plt.show()

# 4. Neighborhood vs SalePrice
# SalePrice distributions vary considerably between the 25 neighborhoods.
# Some neighborhoods have much higher median prices than others,
# showing that Neighborhood can be an important predictive feature.


def plot_year_vs_price(df):
    plt.figure(figsize=(10, 6))

    sns.scatterplot(
        data=df,
        x="YearBuilt",
        y="SalePrice"
    )

    plt.title("Year Built vs Sale Price")
    plt.xlabel("Year Built")
    plt.ylabel("Sale Price")

    plt.show()


# 5. YearBuilt vs SalePrice
# YearBuilt ranges approximately from 1872 to 2010.
# Newer houses generally reach higher price levels,
# but older houses can also have high prices.


def plot_correlation_matrix(df):
    plt.figure(figsize=(14, 10))

    correlation = df.select_dtypes(
        include=["int64", "float64"]
    ).corr()

    sns.heatmap(
        correlation,
        cmap="coolwarm",
        center=0
    )

    plt.title("Correlation Matrix")

    plt.show()


# 6. Correlation matrix
# OverallQual, GrLivArea, GarageCars and GarageArea show strong positive
# relationships with SalePrice.
# Some features are also highly correlated with each other,
# such as GarageCars and GarageArea.




def main():
    df = load_data()

    plot_price_distribution(df)
    surface_vs_price(df)
    plot_quality_vs_price(df)
    plot_neighborhood_vs_price(df)
    plot_year_vs_price(df)
    plot_correlation_matrix(df)



if __name__ == "__main__":
    main()



# EDA conclusion
# The price depends on multiple characteristics of the house.
# We will keep relevant features, investigate outliers without automatically
# removing them, and create new features before modeling.