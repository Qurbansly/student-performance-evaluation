import pandas as pd
import numpy as np
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBRegressor


NUMERICAL_FEATURES = [
    "ESCS", "HOMEPOS", "ICTRES", "ICTHOME",
    "BELONG", "BULLIED", "FAMSUP", "TEACHSUP",
    "MATHEFF", "FAMCON", "ANXMAT"
]

CATEGORICAL_FEATURES = [
    "ST004D01T", "ST005Q01JA", "ST006Q01JA",
    "ST006Q02JA", "ST006Q03JA", "ST007Q01JA",
    "WORKPAY", "WORKHOME", "MATHPREF",
    "MATHEASE", "MATHMOT", "MATHPERS"
]

TARGETS = ["PV1MATH", "PV2MATH", "PV3MATH", "PV4MATH", "PV5MATH"]


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    return {
        "r2": round(r2_score(y_test, predictions), 4),
        "mae": round(mean_absolute_error(y_test, predictions), 4),
        "rmse": round(np.sqrt(mean_squared_error(y_test, predictions)), 4)
    }


def main():
    df = pd.read_spss("data/CY08MSP_STU_QQQ.sav")
    df["AVG_MATH"] = df[TARGETS].mean(axis=1)

    selected_columns = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + ["AVG_MATH"]
    df = df[selected_columns].dropna()

    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y = df["AVG_MATH"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERICAL_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES)
        ]
    )

    tuning_results = []

    models = {
        "Random Forest Tuned": {
            "model": RandomForestRegressor(random_state=42, n_jobs=-1),
            "params": {
                "model__n_estimators": [50, 100],
                "model__max_depth": [8, 10, 12],
                "model__min_samples_split": [2, 5],
                "model__min_samples_leaf": [1, 2]
            }
        },

        "XGBoost Tuned": {
            "model": XGBRegressor(
                random_state=42,
                objective="reg:squarederror",
                n_jobs=-1
            ),
            "params": {
                "model__n_estimators": [100, 200],
                "model__max_depth": [3, 4, 5],
                "model__learning_rate": [0.05, 0.1],
                "model__subsample": [0.8, 1.0],
                "model__colsample_bytree": [0.8, 1.0]
            }
        }
    }

    for name, config in models.items():
        print(f"\nTuning {name}...")

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", config["model"])
        ])

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=config["params"],
            n_iter=6,
            cv=3,
            scoring="r2",
            random_state=42,
            n_jobs=-1,
            verbose=2
        )

        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        metrics = evaluate_model(best_model, X_test, y_test)

        result = {
            "model": name,
            "best_cv_r2": round(search.best_score_, 4),
            "test_r2": metrics["r2"],
            "test_mae": metrics["mae"],
            "test_rmse": metrics["rmse"],
            "best_params": search.best_params_
        }

        tuning_results.append(result)

        filename = name.lower().replace(" ", "_") + "_model.pkl"
        joblib.dump(best_model, f"models/{filename}")

        print(result)
        print(f"{name} saved to models/{filename}")

    results_df = pd.DataFrame(tuning_results)
    results_df.to_csv("results/hyperparameter_tuning_results.csv", index=False)

    print("\nHyperparameter tuning completed successfully.")


if __name__ == "__main__":
    main()
