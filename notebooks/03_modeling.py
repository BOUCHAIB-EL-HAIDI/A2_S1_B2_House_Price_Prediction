import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.preprocessing import (
    load_data,
    prepare_data,
    create_preprocessor,
)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Housing Price Prediction")

df = load_data()

X, y = prepare_data(df)


X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

print("Training shape:", X_train.shape)
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
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
    ),
}


results = []


for model_name, model in models.items():

    print("\n" + "=" * 50)
    print(f"Training: {model_name}")
    print("=" * 50)

    
    preprocessor = create_preprocessor(X_train)

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model),
    ])

    
    with mlflow.start_run(run_name=model_name):


        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_valid)


        mae = mean_absolute_error(
            y_valid,
            y_pred,
        )

        mse = mean_squared_error(
            y_valid,
            y_pred,
        )

        rmse = mse ** 0.5

        r2 = r2_score(
            y_valid,
            y_pred,
        )


        print(f"MAE : {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R²  : {r2:.4f}")


        mlflow.log_param(
            "model",
            model_name,
        )

        mlflow.log_param(
            "test_size",
            0.2,
        )

        mlflow.log_param(
            "random_state",
            42,
        )


        mlflow.log_metric(
            "MAE",
            mae,
        )

        mlflow.log_metric(
            "RMSE",
            rmse,
        )

        mlflow.log_metric(
            "R2",
            r2,
        )

       
        # Log model
       

        mlflow.sklearn.log_model(

            pipeline,
            "model",
            serialization_format="cloudpickle",
        )

        # Save result
    
        results.append({
            "Model": model_name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
        })


results_df = pd.DataFrame(results)

print("\n")
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)