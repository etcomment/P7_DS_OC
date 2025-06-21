import gdown
from flask import Flask, request, jsonify
import pandas as pd
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

file_id = "1k0VyQ6CzMXph0uKWeLbZp9MWvw_PNBwu"  # Remplace par ton ID
destination = "test.csv"

# Construction de l'URL compatible gdown
url = f"https://drive.google.com/uc?id={file_id}"
# Téléchargement
gdown.download(url, destination, quiet=False)
print("✅ Fichier données test téléchargé !")

# Charger le modèle
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Chargement de la liste des features
print("Chargement du fichier de données")
df_donnees = pd.read_csv("test.csv", index_col=False)

@app.route("/")
def index():
    return jsonify({"message": "API de scoring bancaire (LightGBM) prête à l'emploi."})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        print("toto")
        id_client = request.form.get("id_client")

        if id_client is None:
            return jsonify({"error": "id_client est requis."}), 400

        ligne_client = df_donnees[df_donnees["SK_ID_CURR"] == int(id_client)]

        if ligne_client.empty:
            return jsonify({"error": f"Aucun client avec l'id {id_client}."}), 404
        first_column = df_donnees.columns[0]
        # Supposons que le modèle prend toutes les colonnes sauf SK_ID_CURR
        features = ligne_client.drop(columns=[first_column,"index","SK_ID_CURR"])

        # Prédiction (proba d'être "défaillant" ou "bon")
        proba = model.predict_proba(features)[0, 1]  # classe 1 (défaut)

        return jsonify({
            "id_client": int(id_client),
            "score": float(proba)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    waitress.serve(app, host="0.0.0.0", port=10000)
    #waitress.serve(app, host="127.0.0.1", port=10000)

