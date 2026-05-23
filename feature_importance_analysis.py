import pandas as pd
import joblib

MODEL_PATH = "models/xgboost_tuned_model.pkl"

model = joblib.load(MODEL_PATH)

preprocessor = model.named_steps["preprocessor"]
xgb_model = model.named_steps["model"]

feature_names = preprocessor.get_feature_names_out()
importances = xgb_model.feature_importances_

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
})

importance_df = importance_df.sort_values(
    by="importance",
    ascending=False
)

importance_df.to_csv("results/feature_importance.csv", index=False)

print(importance_df.head(20))
print("\nFeature importance saved to results/feature_importance.csv")
