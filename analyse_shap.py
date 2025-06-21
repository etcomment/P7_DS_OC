import shap
import pandas as pd
import pickle
import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import os
import numpy as np

os.makedirs("shap_plots", exist_ok=True)

# Charger le modèle
model = joblib.load("model.pkl")

# Charger les données de test (avec target)
test_data = pd.read_csv("train.csv")
X_test = test_data.drop(columns=["TARGET"])
y_test = test_data["TARGET"]
first_column = X_test.columns[0]
X_test = X_test.drop(columns=[first_column,"index","SK_ID_CURR"])
# Charger les données de validation (sans target)
validation_data = pd.read_csv("test.csv")
X_val = validation_data

# Créer l'explainer
explainer = shap.Explainer(model)

# 🔹 Analyse globale
shap_values = explainer(X_test)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.title("SHAP Summary Plot (Global Analysis)")
plt.savefig("shap_plots/global_summary.png", dpi=300, bbox_inches="tight")
plt.close()

# 🔹 Échantillon de données pour analyse locale
sample_indices = [0, 10, 20]
sample_data = X_test.iloc[sample_indices]
sample_shap_values = explainer(sample_data)

# 🔹 Sauvegarde des plots SHAP locaux
for idx in sample_indices:
    expl = shap_values[idx]  # type: shap.Explanation
    viz = shap.plots.force(expl, matplotlib=False, show=False)  # nouvelle API
    shap.save_html(f"shap_plots/force_plot_{idx}.html", viz)

