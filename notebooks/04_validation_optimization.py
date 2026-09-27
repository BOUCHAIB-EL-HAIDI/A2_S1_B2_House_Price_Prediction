import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import mlflow

from sklearn.model_selection import KFold, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from src.preprocessing import (
    load_data,
    prepare_data,
    create_preprocessor,
)


mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Housing Price Prediction")


df = load_data()

X, y = prepare_data(df)

print("X shape:", X.shape)
print("y shape:", y.shape)


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
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
    ),
}


kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)


results = []

for model_name, model in models.items():

    print("\n" + "=" * 60)
    print(f"Cross-validation: {model_name}")
    print("=" * 60)

    preprocessor = create_preprocessor(X)

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model),
    ])

    with mlflow.start_run(run_name=f"KFold_{model_name}"):

        cv_results = cross_validate(
            pipeline,
            X,
            y,
            cv=kf,
            scoring={
                "mae": "neg_mean_absolute_error",
                "rmse": "neg_root_mean_squared_error",
                "r2": "r2",
            },
            return_train_score=False,
            n_jobs=-1,
        )

        mae_scores = -cv_results["test_mae"]
        rmse_scores = -cv_results["test_rmse"]
        r2_scores = cv_results["test_r2"]

        mean_mae = mae_scores.mean()
        mean_rmse = rmse_scores.mean()
        mean_r2 = r2_scores.mean()
        std_r2 = r2_scores.std()

        print("\nFold MAE:")
        print(mae_scores)

        print("\nFold RMSE:")
        print(rmse_scores)

        print("\nFold R²:")
        print(r2_scores)

        print("\nMean MAE:", mean_mae)
        print("Mean RMSE:", mean_rmse)
        print("Mean R²:", mean_r2)
        print("Std R²:", std_r2)

        mlflow.log_param("model", model_name)
        mlflow.log_param("n_splits", 5)
        mlflow.log_param("random_state", 42)

        mlflow.log_metric("Mean_MAE", mean_mae)
        mlflow.log_metric("Mean_RMSE", mean_rmse)
        mlflow.log_metric("Mean_R2", mean_r2)
        mlflow.log_metric("Std_R2", std_r2)

        results.append({
            "Model": model_name,
            "Mean MAE": mean_mae,
            "Mean RMSE": mean_rmse,
            "Mean R2": mean_r2,
            "Std R2": std_r2,
        })


results_df = pd.DataFrame(results)

print("\n")
print("=" * 70)
print("K-FOLD CROSS-VALIDATION COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


gradient_boosting = GradientBoostingRegressor(
    random_state=42
)

preprocessor = create_preprocessor(X)

pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", gradient_boosting),
])


param_grid = {
    "model__n_estimators": [100, 200, 300],
    "model__learning_rate": [0.03, 0.05, 0.1],
    "model__max_depth": [2, 3, 4],
}


with mlflow.start_run(run_name="GridSearch_GradientBoosting"):

    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=kf,
        scoring={
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
        },
        refit="r2",
        n_jobs=-1,
        return_train_score=False,
    )

    grid_search.fit(X, y)

    print("\n")
    print("=" * 70)
    print("GRID SEARCH RESULTS")
    print("=" * 70)

    print("\nBest parameters:")
    print(grid_search.best_params_)

    best_index = grid_search.best_index_

    best_mae = -grid_search.cv_results_["mean_test_mae"][best_index]
    best_rmse = -grid_search.cv_results_["mean_test_rmse"][best_index]
    best_r2 = grid_search.cv_results_["mean_test_r2"][best_index]

    print("\nOptimized Mean MAE:")
    print(best_mae)

    print("\nOptimized Mean RMSE:")
    print(best_rmse)

    print("\nOptimized Mean R²:")
    print(best_r2)

    mlflow.log_param("model", "GradientBoosting")
    mlflow.log_param("cv", 5)
    mlflow.log_param("refit", "r2")

    mlflow.log_params(grid_search.best_params_)

    mlflow.log_metric("Optimized_MAE", best_mae)
    mlflow.log_metric("Optimized_RMSE", best_rmse)
    mlflow.log_metric("Optimized_R2", best_r2)




# J'ai choisi ces hyperparamètres car ils contrôlent directement la complexité
# et l'apprentissage du Gradient Boosting.
# n_estimators contrôle le nombre d'arbres,
# learning_rate contrôle la contribution de chaque arbre,
# et max_depth contrôle la profondeur des arbres.
# J'ai testé plusieurs valeurs afin de rechercher un compromis
# entre performance et complexité.