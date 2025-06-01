from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import requests
import pickle

app = Flask(__name__)

# Chargement du modèle
#model = joblib.load('model.pkl')

url = "https://github.com/etcomment/P7_DS_OC/raw/refs/heads/API/mlartifacts/0/1dc35f31d41a47f5a8698f78ef00ea0f/artifacts/model/model.pkl"
response = requests.get(url)

with open("model.pkl", "wb") as f:
    f.write(response.content)

# Charger le modèle
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Chargement de la liste des features
with open("features_used.txt", "r") as f:
    all_features = [line.strip() for line in f if line.strip()]

@app.route("/")
def index():
    return jsonify({"message": "API de scoring bancaire (LightGBM) prête à l'emploi."})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = request.get_json()

        # Construction du DataFrame d'une seule ligne
        input_df = pd.DataFrame([input_data])

        # Ajout des colonnes manquantes avec NaN
        for col in all_features:
            if col not in input_df.columns:
                input_df[col] = np.nan

        # Réordonnancement des colonnes selon le modèle
        input_df = input_df[all_features]

        # Forcer le typage des colonnes catégorielles si nécessaire :
        # Ex : input_df['csp'] = input_df['csp'].astype('category')

        # Prédiction
        score = model.predict_proba(input_df)[0, 1]

        return jsonify({'score': float(score)})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(debug=False)
