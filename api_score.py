from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Chargement du modèle
model = joblib.load('model.pkl')

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
    app.run(debug=True)
