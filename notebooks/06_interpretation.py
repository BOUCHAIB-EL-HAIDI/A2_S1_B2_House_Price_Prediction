import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parents[1])
)

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from src.preprocessing import (
    prepare_data,
    create_preprocessor,
)


df = pd.read_csv(
    "data/processed/train_features_no_outliers.csv"
)

X, y = prepare_data(df)

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)


models = {
    "LinearRegression": LinearRegression(),

    "RandomForest": RandomForestRegressor(
        n_estimators=900,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    ),

    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
    ),
}


pipelines = {}


for name, model in models.items():

    print("\n" + "=" * 70)
    print(f"TRAINING: {name}")
    print("=" * 70)

    preprocessor = create_preprocessor(X_train)

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model),
    ])

    pipeline.fit(
        X_train,
        y_train
    )

    pipelines[name] = pipeline

    print("Model trained successfully.")


def get_feature_mapping(preprocessor, X):

    feature_mapping = {}

    numeric_columns = X.select_dtypes(
        include=["int64", "float64"]
    ).columns

    categorical_columns = X.select_dtypes(
        include=["object", "str"]
    ).columns

    for column in numeric_columns:
        transformed_name = f"numerical__{column}"
        feature_mapping[transformed_name] = column

    for column in categorical_columns:
        prefix = f"categorical__{column}_"

        for feature_name in preprocessor.get_feature_names_out():
            if feature_name.startswith(prefix):
                feature_mapping[feature_name] = column

    return feature_mapping


gradient_pipeline = pipelines["GradientBoosting"]

preprocessor = gradient_pipeline.named_steps[
    "preprocessing"
]

model = gradient_pipeline.named_steps[
    "model"
]

feature_names = preprocessor.get_feature_names_out()

feature_mapping = get_feature_mapping(
    preprocessor,
    X_train
)


print("\n")
print("=" * 70)
print("FEATURE NAMES AFTER PREPROCESSING")
print("=" * 70)

print(
    f"Number of transformed features: {len(feature_names)}"
)


def aggregate_importances(
    feature_names,
    values,
    feature_mapping
):

    importance_df = pd.DataFrame({
        "TransformedFeature": feature_names,
        "Importance": values,
    })

    importance_df["OriginalFeature"] = (
        importance_df["TransformedFeature"]
        .map(feature_mapping)
    )

    importance_df = importance_df.dropna(
        subset=["OriginalFeature"]
    )

    aggregated = (
        importance_df
        .groupby("OriginalFeature")["Importance"]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    return aggregated


gradient_importances = aggregate_importances(
    feature_names,
    model.feature_importances_,
    feature_mapping
)


print("\n")
print("=" * 70)
print("GRADIENT BOOSTING - FEATURE IMPORTANCE")
print("=" * 70)

print(
    gradient_importances
    .head(15)
    .to_string()
)


plt.figure(figsize=(10, 6))

gradient_importances.head(15).sort_values().plot(
    kind="barh"
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title(
    "Top 15 Feature Importance - Gradient Boosting"
)

plt.tight_layout()
plt.show()


random_forest_pipeline = pipelines["RandomForest"]

preprocessor = random_forest_pipeline.named_steps[
    "preprocessing"
]

model = random_forest_pipeline.named_steps[
    "model"
]

feature_names = preprocessor.get_feature_names_out()

feature_mapping = get_feature_mapping(
    preprocessor,
    X_train
)

random_forest_importances = aggregate_importances(
    feature_names,
    model.feature_importances_,
    feature_mapping
)


print("\n")
print("=" * 70)
print("RANDOM FOREST - FEATURE IMPORTANCE")
print("=" * 70)

print(
    random_forest_importances
    .head(15)
    .to_string()
)


plt.figure(figsize=(10, 6))

random_forest_importances.head(15).sort_values().plot(
    kind="barh"
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title(
    "Top 15 Feature Importance - Random Forest"
)

plt.tight_layout()
plt.show()


linear_pipeline = pipelines["LinearRegression"]

preprocessor = linear_pipeline.named_steps[
    "preprocessing"
]

model = linear_pipeline.named_steps[
    "model"
]

feature_names = preprocessor.get_feature_names_out()

feature_mapping = get_feature_mapping(
    preprocessor,
    X_train
)

coefficients = pd.DataFrame({
    "TransformedFeature": feature_names,
    "Coefficient": model.coef_,
})

coefficients["OriginalFeature"] = (
    coefficients["TransformedFeature"]
    .map(feature_mapping)
)

coefficients = coefficients.dropna(
    subset=["OriginalFeature"]
)

coefficients["AbsoluteCoefficient"] = (
    coefficients["Coefficient"].abs()
)

aggregated_coefficients = (
    coefficients
    .groupby("OriginalFeature")[
        "AbsoluteCoefficient"
    ]
    .sum()
    .sort_values(
        ascending=False
    )
)


print("\n")
print("=" * 70)
print("LINEAR REGRESSION - COEFFICIENTS")
print("=" * 70)

print(
    aggregated_coefficients
    .head(15)
    .to_string()
)


plt.figure(figsize=(10, 6))

aggregated_coefficients.head(15).sort_values().plot(
    kind="barh"
)

plt.xlabel("Somme des valeurs absolues des coefficients")
plt.ylabel("Feature")
plt.title(
    "Top 15 Features - Linear Regression"
)

plt.tight_layout()
plt.show()


positive_coefficients = (
    coefficients[
        coefficients["Coefficient"] > 0
    ]
    .sort_values(
        by="Coefficient",
        ascending=False
    )
    .head(10)
)

negative_coefficients = (
    coefficients[
        coefficients["Coefficient"] < 0
    ]
    .sort_values(
        by="Coefficient"
    )
    .head(10)
)


print("\n")
print("=" * 70)
print("TOP 10 COEFFICIENTS POSITIFS")
print("=" * 70)

print(
    positive_coefficients[
        [
            "TransformedFeature",
            "OriginalFeature",
            "Coefficient",
        ]
    ].to_string(
        index=False
    )
)


print("\n")
print("=" * 70)
print("TOP 10 COEFFICIENTS NEGATIFS")
print("=" * 70)

print(
    negative_coefficients[
        [
            "TransformedFeature",
            "OriginalFeature",
            "Coefficient",
        ]
    ].to_string(
        index=False
    )
)


comparison = pd.DataFrame({
    "GradientBoosting": gradient_importances,
    "RandomForest": random_forest_importances,
    "LinearRegression": aggregated_coefficients,
})

comparison = comparison.fillna(0)

comparison["GradientBoosting_Rank"] = (
    comparison["GradientBoosting"]
    .rank(
        ascending=False,
        method="min"
    )
)

comparison["RandomForest_Rank"] = (
    comparison["RandomForest"]
    .rank(
        ascending=False,
        method="min"
    )
)

comparison["LinearRegression_Rank"] = (
    comparison["LinearRegression"]
    .rank(
        ascending=False,
        method="min"
    )
)

comparison = comparison.sort_values(
    by="GradientBoosting",
    ascending=False
)


print("\n")
print("=" * 70)
print("COMPARAISON DES IMPORTANCES")
print("=" * 70)

print(
    comparison.head(15).to_string()
)


top_features = comparison.head(10).index.tolist()

print("\n")
print("=" * 70)
print("TOP 10 FEATURES COMMUNES")
print("=" * 70)

for position, feature in enumerate(
    top_features,
    start=1
):
    print(
        f"{position}. {feature}"
    )




# L'interprétation des modèles montre que OverallQual et TotalSF sont les
# caractéristiques les plus importantes pour les modèles Gradient Boosting
# et Random Forest. D'autres variables comme TotalBathrooms, HouseAge,
# BsmtFinSF1, BsmtQual et GrLivArea contribuent également aux prédictions.
# 
# Les résultats sont globalement cohérents avec les observations réalisées
# lors de l'EDA, notamment concernant l'importance de la qualité globale,
# de la surface et des caractéristiques du logement.
# 
# La régression linéaire permet également d'identifier l'influence des
# variables numériques et des catégories après encodage. Cependant,
# l'interprétation des coefficients catégoriels doit tenir compte de la
# catégorie de référence utilisée par le OneHotEncoder.
# 
# L'analyse confirme ainsi que les caractéristiques identifiées pendant
# l'EDA jouent un rôle important dans les prédictions des modèles.