# API de Scoring Bancaire

Ce dépôt contient une API développée en Flask permettant de produire un score de probabilité de défaut pour un client bancaire à partir de caractéristiques d'entrée. L'API repose sur un modèle de classification entraîné (LightGBM), et est accompagnée de tests unitaires pour en assurer la robustesse.

---

## 1. Objectifs

- Exposer une API RESTful pour prédire la probabilité de défaut client à partir d'un identifiant.
- Intégrer une logique de téléchargement automatique du modèle et des données depuis Google Drive.
- Fournir un ensemble de tests unitaires vérifiant la stabilité du système.
- Permettre une exécution en environnement de production via Waitress.

---

## 2. Installation

### Prérequis

- Python 3.8+
- pip

### Étapes

1. Cloner le dépôt :

```bash
git clone https://github.com/<utilisateur>/<nom-du-repo>.git
cd <nom-du-repo>
```

2. Créer un environnement virtuel et activer :

```bash
python -m venv env
source env/bin/activate  # Windows : env\Scripts\activate
```

3. Installer les dépendances :

```bash
pip install -r requirements.txt
```

---

## 3. Lancement de l’API

L'API se lance avec le fichier principal `api_score.py`.

```bash
python api_score.py
```

Le serveur est exposé par défaut sur le port `10000`.

---

## 4. Utilisation de l’API

### Endpoint : `/predict` (méthode POST)

**Paramètre attendu** : `id_client` (form-data)

Exemple avec `curl` :

```bash
curl -X POST http://localhost:10000/predict -F id_client=100002
```

Réponse attendue :

```json
{
  "id_client": 100002,
  "score": 0.231984
}
```

---

## 5. Tests unitaires

Les tests sont définis dans le fichier `tu_api_score.py`.

### Exécution :

```bash
pytest tu_api_score.py
```

Les cas testés incluent :

- Appel au endpoint `/`
- Absence du paramètre `id_client`
- Utilisation d’un identifiant inconnu
- Réponse attendue avec un identifiant valide

---

## 6. Dépendances

Fichier `requirements.txt` typique :

```
Flask
waitress
pandas
numpy
scikit-learn
gdown
pytest
```

---

## 7. Contribution

Les contributions sont les bienvenues. Pour contribuer :

1. Créer un fork du projet
2. Créer une branche dédiée : `feature/ma-fonctionnalite`
3. Vérifier que tous les tests passent (`pytest`)
4. Proposer une pull request documentée

---

## 8. Contact

Pour toute question, remarque ou problème technique, merci de créer une issue sur le dépôt GitHub.
