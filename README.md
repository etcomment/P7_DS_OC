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

- Python 3.10+
- pip
- Droit d'accès au repo
- Venv

### Étapes

1. Cloner le dépôt :

```bash
git https://github.com/etcomment/P7_DS_OC.git
cd P7_DS_OC
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
python GUILLAUME_Stiven_1_API_052025.py
```

Le serveur est exposé par défaut sur le port `10000`, en conformité avec les précos de render.com

Il est cependant possible de modifier si besoin ce parametrage 

---

## 4. Utilisation de l’API

### Endpoint : `/predict` (méthode POST)

**Paramètre attendu** : `id_client` (form-data)

Exemple avec `curl` :

```bash
curl -X POST https://p7-ds-oc.onrender.com/predict -F id_client=100002
```

Réponse attendue :

```json
{
  "id_client": 100001,
  "score": 0.059778321535366476
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

Voir le fichier `requirements.txt`  :


---

## 7. Contribution

Les contributions sont les bienvenues. Pour contribuer :

1. Créer un fork du projet
2. Créer une branche dédiée : `feature/ma-fonctionnalite`
3. Vérifier que tous les tests passent (`pytest`)
4. Proposer une pull request documentée

---

## 8. Todo list

Vous ne savez pas comment contribuer ? Voici une liste d'éléments qu'on a besoin pour l'API :

- Retour de Shap
- Completer / Fiabiliser les TU

---
## 9. Contact

Pour toute question, remarque ou problème technique, merci de créer une issue sur le dépôt GitHub.
