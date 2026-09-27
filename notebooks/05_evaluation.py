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

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from src.preprocessing import (
    prepare_data,
    create_preprocessor,
)


df = pd.read_csv(
    "data/processed/train_features_no_outliers.csv"
)

X, y = prepare_data(df)

print("X shape:", X.shape)
print("y shape:", y.shape)


X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

print("\nTraining shape:", X_train.shape)
print("Validation shape:", X_valid.shape)


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


results = []
predictions = {}


for name, model in models.items():

    print("\n" + "=" * 60)
    print(f"Training: {name}")
    print("=" * 60)

    preprocessor = create_preprocessor(X_train)

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model),
    ])

    pipeline.fit(
        X_train,
        y_train
    )

    y_pred = pipeline.predict(
        X_valid
    )

    predictions[name] = y_pred

    mae = mean_absolute_error(
        y_valid,
        y_pred
    )

    rmse = mean_squared_error(
        y_valid,
        y_pred
    ) ** 0.5

    r2 = r2_score(
        y_valid,
        y_pred
    )

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    })

    print(f"MAE : {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²  : {r2:.4f}")


results_df = pd.DataFrame(results)

print("\n")
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


y_pred = predictions["GradientBoosting"]


plt.figure(figsize=(8, 6))

plt.scatter(
    y_valid,
    y_pred,
    alpha=0.6
)

plt.xlabel("Prix réel")
plt.ylabel("Prix prédit")
plt.title(
    "Prix réel vs prix prédit - Gradient Boosting"
)

min_price = min(
    y_valid.min(),
    y_pred.min()
)

max_price = max(
    y_valid.max(),
    y_pred.max()
)

plt.plot(
    [min_price, max_price],
    [min_price, max_price],
    linestyle="--"
)

plt.tight_layout()
plt.show()


errors = y_valid - y_pred

absolute_errors = errors.abs()

evaluation_df = pd.DataFrame({
    "ActualPrice": y_valid,
    "PredictedPrice": y_pred,
    "Error": errors,
    "AbsoluteError": absolute_errors,
}, index=y_valid.index)


print("\n")
print("=" * 70)
print("ERROR ANALYSIS")
print("=" * 70)

print(
    evaluation_df.describe()
)


plt.figure(figsize=(8, 6))

plt.hist(
    errors,
    bins=30
)

plt.xlabel("Erreur (prix réel - prix prédit)")
plt.ylabel("Nombre de logements")
plt.title(
    "Distribution des erreurs - Gradient Boosting"
)

plt.axvline(
    0,
    linestyle="--"
)

plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 6))

plt.scatter(
    y_pred,
    errors,
    alpha=0.6
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel("Prix prédit")
plt.ylabel("Résidu")
plt.title(
    "Analyse des résidus - Gradient Boosting"
)

plt.tight_layout()
plt.show()


largest_errors = (
    evaluation_df
    .sort_values(
        by="AbsoluteError",
        ascending=False
    )
    .head(10)
)

print("\n")
print("=" * 70)
print("10 PLUS GRANDES ERREURS DE PRÉDICTION")
print("=" * 70)

print(
    largest_errors.to_string(
        index=False
    )
)


error_indices = largest_errors.index

error_features = X_valid.loc[
    error_indices
].copy()

error_features["ActualPrice"] = y_valid.loc[
    error_indices
]

error_features["PredictedPrice"] = (
    predictions["GradientBoosting"][
        X_valid.index.get_indexer(error_indices)
    ]
)

error_features["Error"] = (
    error_features["ActualPrice"]
    - error_features["PredictedPrice"]
)

error_features["AbsoluteError"] = (
    error_features["Error"].abs()
)


columns_to_display = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "GarageArea",
    "TotalBsmtSF",
    "1stFlrSF",
    "YearBuilt",
    "YearRemodAdd",
    "Neighborhood",
    "KitchenQual",
    "ExterQual",
    "BsmtQual",
    "GarageFinish",
    "ActualPrice",
    "PredictedPrice",
    "Error",
    "AbsoluteError",
]

available_columns = [
    column
    for column in columns_to_display
    if column in error_features.columns
]

missing_columns = [
    column
    for column in columns_to_display
    if column not in error_features.columns
]

print("\nColonnes disponibles:")
print(available_columns)

print("\nColonnes absentes:")
print(missing_columns)

print("\n")
print("=" * 70)
print("CARACTÉRISTIQUES DES 10 PLUS GRANDES ERREURS")
print("=" * 70)

print(
    error_features[
        available_columns
    ].to_string()
)


# L'analyse des erreurs montre que les plus grandes erreurs apparaissent principalement
# sur des logements dont le prix réel est atypique par rapport à leurs caractéristiques.
# Le modèle peut sous-estimer certains logements de bonne qualité ou surestimer des
# logements présentant une grande surface mais des caractéristiques qualitatives plus faibles.
# Ces erreurs peuvent également être liées à des informations non retenues lors de la sélection
# des variables. Cela montre que même un modèle avec de bonnes performances globales peut
# rencontrer des difficultés sur certaines observations particulières.