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
user_id = columns[1].text_input("id_client *", max_chars=6)

user_input["id_client"] = user_id


# ---------- BOUTON POUR ENVOYER LA REQUÊTE ----------
if st.button("✅ Vérifier l’éligibilité"):

    # Vérifier que les champs obligatoires sont bien fournis
    if "id_client" not in user_input:
        st.error("Veuillez renseigner tous les champs obligatoires.")
    else:
        try:
            # Appel à l'API Flask (local ou déployée)
            print(user_input)
            api_url = "https://p7-ds-oc.onrender.com/predict"
            #api_url = "https://127.0.0.1:10000/predict"
            response = requests.post(api_url, user_input)

            if response.status_code == 200:
                result = response.json()
                score = result["score"]

                st.markdown(f"### 🎯 Score obtenu : **{score:.2%}** (Seuil d'inegibilité : 47%)")

                if score <= 0.47:
                    st.success("✅ Éligible")
                else:
                    st.error("❌ Non éligible")
            else:
                st.error(f"Erreur API : {response.status_code} - {response.text}")

        except Exception as e:
            st.error(f"Erreur de communication avec l'API : {e}")
