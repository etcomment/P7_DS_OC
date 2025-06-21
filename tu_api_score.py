import pytest
from GUILLAUME_Stiven_1_API_052025 import app, df_donnees

# Création d'un client de test Flask
@pytest.fixture
def client():
    app.testing = True
    with app.test_client() as client:
        yield client

# Exemple de récupération d’un ID client existant
@pytest.fixture
def valid_id():
    return int(df_donnees["SK_ID_CURR"].dropna().sample(1).values[0])

# ✅ Test de l’endpoint racine
def test_index(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "message" in res.get_json()

# ❌ Test sans id_client
def test_predict_missing_id(client):
    res = client.post("/predict", data={})  # pas d'id_client
    assert res.status_code == 400
    assert "error" in res.get_json()

# ❌ Test avec un id_client inexistant
def test_predict_invalid_id(client):
    res = client.post("/predict", data={"id_client": 999999999})
    assert res.status_code == 404
    assert "error" in res.get_json()

# ✅ Test avec un id_client valide
def test_predict_valid_id(client, valid_id):
    res = client.post("/predict", data={"id_client": valid_id})
    json_data = res.get_json()

    assert res.status_code == 200
    assert "score" in json_data
    assert isinstance(json_data["score"], float)
    assert 0.0 <= json_data["score"] <= 1.0