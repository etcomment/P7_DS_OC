import streamlit as st
import pandas as pd
import datetime
import requests
import numpy as np

# ---------- CONFIGURATION ----------
st.set_page_config(page_title="Vérification d'éligibilité", layout="wide")

# ---------- TITRE ----------
st.title("🧾 Scoring Bancaire - Vérification d'Éligibilité")

# ---------- CHARGEMENT DES FEATURES ----------
@st.cache_data
def load_features():
    with open("features_used.txt", "r") as f:
        return [line.strip() for line in f if line.strip()]

all_features = load_features()

# ---------- FONCTION POUR CALCULER DAYS_BIRTH ----------
def age_en_nbjours(date_naissance):
    today = datetime.date.today()
    delta = today - date_naissance
    return -delta.days  # LightGBM attend un nombre négatif pour DAYS_BIRTH

# ---------- FORMULAIRE ----------
st.subheader("📋 Informations du client")
user_input = {}
columns = st.columns(3)

# Champs obligatoires
date_naissance = columns[0].date_input("Date de naissance *", value=datetime.date(1990, 1, 1))
payment_rate = columns[1].number_input("PAYMENT_RATE *", min_value=0.0, format="%.6f")

user_input["DAYS_BIRTH"] = age_en_nbjours(date_naissance)
user_input["PAYMENT_RATE"] = payment_rate

# Champs facultatifs
st.markdown("### ⚙️ Autres données facultatives")
with st.expander("Afficher / Modifier les autres paramètres"):
    for feature in all_features:
        if feature in ["DAYS_BIRTH", "PAYMENT_RATE"]:
            continue  # déjà traités

        # Création de champs dynamiques selon le nom
        if feature.startswith("FLAG_") or feature.startswith("NAME_") or feature.startswith("OCCUPATION_"):
            val = st.selectbox(f"{feature}", ["0", "1", "non renseigné"], index=2)
            if val != "non renseigné":
                user_input[feature] = int(val)
        elif feature.startswith("CODE_") or feature.startswith("WEEKDAY_") or feature.endswith("_MODE"):
            val = st.text_input(f"{feature} (texte libre)")
            if val:
                user_input[feature] = val
        else:
            val = st.text_input(f"{feature}", placeholder="(optionnel)")
            if val:
                try:
                    user_input[feature] = float(val)
                except ValueError:
                    st.warning(f"⚠️ '{feature}' ignoré (valeur non numérique)")

# ---------- BOUTON POUR ENVOYER LA REQUÊTE ----------
if st.button("✅ Vérifier l’éligibilité"):

    # Vérifier que les champs obligatoires sont bien fournis
    if "PAYMENT_RATE" not in user_input or "DAYS_BIRTH" not in user_input:
        st.error("Veuillez renseigner tous les champs obligatoires.")
    else:
        try:
            # Appel à l'API Flask (local ou déployée)
            api_url = "http://127.0.0.1:5001/predict"  # ⚠️ changer cette URL si déployé sur Azure
            response = requests.post(api_url, json=user_input)

            if response.status_code == 200:
                result = response.json()
                score = result["score"]

                st.markdown(f"### 🎯 Score obtenu : **{score:.2%}**")

                if score >= 0.5:
                    st.success("✅ Éligible")
                else:
                    st.error("❌ Non éligible")
            else:
                st.error(f"Erreur API : {response.status_code} - {response.text}")

        except Exception as e:
            st.error(f"Erreur de communication avec l'API : {e}")
