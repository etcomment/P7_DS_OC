import streamlit as st
import pandas as pd
import datetime
import requests
import numpy as np

# ---------- CONFIGURATION ----------
st.set_page_config(page_title="Vérification d'éligibilité", layout="wide")

# ---------- TITRE ----------
st.title("🧾 Scoring Bancaire - Vérification d'Éligibilité")

# ---------- FORMULAIRE ----------
st.subheader("📋 Informations du client")
user_input = {}
columns = st.columns(3)

# Champs obligatoires
user_id = columns[1].text_input("SK_ID_CURR *", max_chars=10)

user_input["SK_ID_CURR"] = user_id


# ---------- BOUTON POUR ENVOYER LA REQUÊTE ----------
if st.button("✅ Vérifier l’éligibilité"):

    # Vérifier que les champs obligatoires sont bien fournis
    if "SK_ID_CURR" not in user_input:
        st.error("Veuillez renseigner tous les champs obligatoires.")
    else:
        try:
            # Appel à l'API Flask (local ou déployée)
            api_url = "http://127.0.0.1:5001/predict"
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
