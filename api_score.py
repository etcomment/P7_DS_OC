from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import requests
import pickle
import waitress

app = Flask(__name__)

file_id = '1Y8qjL3nPUO7oAAU06LS4_z4DDkomo4BI'  # à extraire de l'URL
download_url = f'https://drive.google.com/uc?export=download&id={file_id}'

response = requests.get(download_url)
with open("model.pkl", "wb") as f:
    f.write(response.content)

print("✅ Fichier modele téléchargé !")

file_id = '1B2E6jfa1DZVdz5yGDeBitJ_0cS-qeUR2'  # à extraire de l'URL
download_url = f'https://drive.google.com/uc?export=download&id={file_id}'

response = requests.get(download_url)
with open("features_used.txt", "wb") as f:
    f.write(response.content)

print("✅ Fichier features téléchargé !")

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
        input_df['CC_NAME_CONTRACT_STATUS_Active_MIN'] = input_df['CC_NAME_CONTRACT_STATUS_Active_MIN'].astype('category')
        input_df['CC_NAME_CONTRACT_STATUS_Active_MAX'] = input_df['CC_NAME_CONTRACT_STATUS_Active_MAX'].astype('category')
        input_df['CC_NAME_CONTRACT_STATUS_Approved_MIN'] = input_df['CC_NAME_CONTRACT_STATUS_Approved_MIN'].astype('category')
        input_df['CC_NAME_CONTRACT_STATUS_Approved_MAX'] = input_df['CC_NAME_CONTRACT_STATUS_Approved_MAX'].astype('category')
        input_df['CC_NAME_CONTRACT_STATUS_Completed_MIN'] = input_df['CC_NAME_CONTRACT_STATUS_Completed_MIN'].astype('category')
        input_df['CC_NAME_CONTRACT_STATUS_Completed_MAX'] = input_df['CC_NAME_CONTRACT_STATUS_Completed_MAX'].astype('category')
        input_df['CC_NAME_CONTRACT_STATUS_Demand_MIN'] = input_df['CC_NAME_CONTRACT_STATUS_Demand_MIN'].astype('category')
        input_df['CC_NAME_CONTRACT_STATUS_Demand_MAX'] = input_df['CC_NAME_CONTRACT_STATUS_Demand_MAX'].astype('category')
        input_df['CC_NAME_CONTRACT_STATUS_Refused_MIN'] = input_df['CC_NAME_CONTRACT_STATUS_Refused_MIN'].astype('category')
        input_df['CC_NAME_CONTRACT_STATUS_Refused_MAX'] = input_df['CC_NAME_CONTRACT_STATUS_Refused_MAX'].astype('category')
        input_df['CC_NAME_CONTRACT_STATUS_Sent_proposal_MIN'] = input_df['CC_NAME_CONTRACT_STATUS_Sent_proposal_MIN'].astype('category')
        input_df['CC_NAME_CONTRACT_STATUS_Sent_proposal_MAX'] = input_df['CC_NAME_CONTRACT_STATUS_Sent_proposal_MAX'].astype('category')
        input_df['CC_NAME_CONTRACT_STATUS_Signed_MIN'] = input_df['CC_NAME_CONTRACT_STATUS_Signed_MIN'].astype('category')
        input_df['CC_NAME_CONTRACT_STATUS_Signed_MAX'] = input_df['CC_NAME_CONTRACT_STATUS_Signed_MAX'].astype('category')
        input_df['CC_NAME_CONTRACT_STATUS_nan_MIN'] = input_df['CC_NAME_CONTRACT_STATUS_nan_MIN'].astype('category')
        input_df['CC_NAME_CONTRACT_STATUS_nan_MAX'] = input_df['CC_NAME_CONTRACT_STATUS_nan_MAX'].astype('category')


        # Prédiction
        score = model.predict_proba(input_df)[0, 1]

        return jsonify({'score': float(score)})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    waitress.serve(app, host="0.0.0.0", port=10000)
