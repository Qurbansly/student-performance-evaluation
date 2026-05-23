import pandas as pd

comparison = pd.read_csv("results/regression_model_comparison_results.csv")
tuning = pd.read_csv("results/hyperparameter_tuning_results.csv")

tuning = tuning.rename(columns={
    "test_r2": "r2",
    "test_mae": "mae",
    "test_rmse": "rmse"
})

tuning = tuning[["model", "r2", "mae", "rmse", "best_cv_r2", "best_params"]]

comparison["best_cv_r2"] = comparison["cv_r2_mean"]
comparison["best_params"] = "Default parameters"

final_results = pd.concat([
    comparison[["model", "r2", "mae", "rmse", "best_cv_r2", "best_params"]],
    tuning
])

final_results.to_csv("results/final_model_comparison.csv", index=False)

print(final_results)
print("\nFinal model comparison saved to results/final_model_comparison.csv")
