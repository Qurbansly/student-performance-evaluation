import pandas as pd
import matplotlib.pyplot as plt

results = pd.read_csv("results/final_model_comparison.csv")
features = pd.read_csv("results/feature_importance.csv")

plt.figure(figsize=(10, 6))
plt.bar(results["model"], results["r2"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("R² Score")
plt.title("Model Comparison by R² Score")
plt.tight_layout()
plt.savefig("results/model_r2_comparison.png")
plt.close()

plt.figure(figsize=(10, 6))
plt.bar(results["model"], results["rmse"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("RMSE")
plt.title("Model Comparison by RMSE")
plt.tight_layout()
plt.savefig("results/model_rmse_comparison.png")
plt.close()

top_features = features.head(15)

plt.figure(figsize=(10, 6))
plt.barh(top_features["feature"], top_features["importance"])
plt.xlabel("Importance")
plt.title("Top 15 Feature Importances - Tuned XGBoost")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("results/feature_importance.png")
plt.close()

print("Graphs saved successfully in results folder.")
