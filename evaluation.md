## Définir la stratégie d’élaboration d’un modèle d’apprentissage supervisé

Sélectionner et entraîner des modèles adaptés à une problématique métier afin de réaliser une analyse prédictive.

- [X] **CE1** Les variables catégorielles identifiées ont été transformées en fonction du besoin (par exemple via OneHotEncoder ou TargetEncoder).  
- [X] **CE2** Le candidat a créé de nouvelles variables à partir de variables existantes.  
- [ ] **CE3** Le candidat a réalisé des transformations mathématiques lorsque c'est requis pour transformer les distributions de variables.  
- [ ] **CE4** Le candidat a normalisé les variables lorsque c'est requis.  
- [X] **CE5** Le candidat a défini sa stratégie d’élaboration d’un modèle pour répondre à un besoin métier.  
    - Cela signifie dans ce projet que l’étudiant a présenté son approche méthodologique de modélisation dans son support de présentation pendant la soutenance et est capable de répondre à des questions à ce sujet.  
- [X] **CE6** Le candidat a choisi la ou les variables cibles pertinentes.  
- [ ] **CE7** Le candidat a vérifié qu'il n’y a pas de problème de data leakage.  
- [X] **CE8** Le candidat a testé plusieurs algorithmes de façon cohérente, en partant des plus simples vers les plus complexes (au minimum un linéaire et un non linéaire).  

---

## Évaluer les performances des modèles d’apprentissage supervisé

Adapter les paramètres afin de choisir le modèle le plus performant pour la problématique métier.

- [X] **CE1** Le candidat a choisi une métrique adaptée pour évaluer la performance d'un algorithme (R2, RMSE, accuracy, AUC, etc.).  
    - Dans le cadre de ce projet : mise en œuvre d’un score métier tenant compte du coût des faux positifs/négatifs.  
- [X] **CE2** Le candidat a exploré d'autres indicateurs de performance que le score pour comprendre les résultats (coefficients, visualisations, temps de calcul...).  
- [X] **CE3** Le candidat a séparé les données en train/test pour évaluer correctement et détecter l'overfitting.  
- [X] **CE4** Le candidat a mis en place un modèle de référence (dummyRegressor ou dummyClassifier).  
- [X] **CE5** Le candidat a pris en compte l’éventuel déséquilibre des classes.  
- [X] **CE6** Le candidat a optimisé les hyperparamètres des algorithmes.  
- [X] **CE7** Le candidat a mis en place une validation croisée (GridSearchCV, RandomizedSearchCV, etc.).  
    - Cela implique :  
        - Une cross-validation du dataset train.  
        - Un test initial d’hyperparamètres pour chaque algorithme.  
        - Un affinement pour l’algorithme final.  
        - Toute AUC > 0.82 dans GridSearchCV sans justification = projet invalide.  
- [ ] **CE8** Le candidat a présenté les résultats du plus simple au plus complexe et justifié le choix final.  
- [X] **CE9** Le candidat a analysé l’importance des variables globalement et localement.

---

## Définir et mettre en œuvre un pipeline d’entraînement

Centralisation du stockage des modèles, formalisation des résultats et industrialisation.

- [ ] **CE1** Le candidat a mis en œuvre un pipeline d’entraînement reproductible.  
- [ ] **CE2** Le candidat a sérialisé et stocké les modèles dans un registre centralisé.  
- [ ] **CE3** Le candidat a formalisé les mesures et résultats de chaque expérimentation pour les analyser et comparer.

---

## Mettre en œuvre un logiciel de version de code

Assurer l’intégration et la diffusion du modèle auprès de collaborateurs.

- [X] **CE1** Le candidat a créé un dossier Git avec tous les scripts du projet et l’a partagé sur GitHub.  
- [X] **CE2** Le candidat a présenté un historique de modifications avec au moins trois versions distinctes.  
- [X] **CE3** Le candidat a tenu à jour la liste des packages et versions utilisés.  
- [ ] **CE4** Le candidat a rédigé un fichier introductif expliquant le projet et la structure.  
- [X] **CE5** Le candidat a commenté les scripts pour faciliter la collaboration.

---

## Déploiement continu d'un moteur d’inférence

Déploiement d’un modèle sous forme d’API sur une plateforme Cloud.

- [ ] **CE1** Le candidat a défini un pipeline de déploiement continu.  
- [ ] **CE2** Le modèle a été déployé sous forme d’API (Flask, etc.) avec retour de prédiction.  
- [ ] **CE3** Le pipeline de déploiement déploie l’API sur un serveur Cloud.  
- [ ] **CE4** Des tests unitaires automatisés ont été mis en place (ex : pyTest).  
- [X] **CE5** L’API est indépendante de l’application qui l’utilise.

---

## Suivi de la performance d’un modèle en production

Maintenance et suivi pour garantir des prédictions performantes dans le temps.

- [ ] **CE1** Le candidat a défini une stratégie de suivi de performance du modèle.  
    - Par exemple, analyse de data drift entre train et test.  
- [ ] **CE2** Le candidat a mis en place un stockage des événements et une gestion d’alerte en cas de dégradation.  
    - Simulation de drift dans un notebook, analyse via `evidently`, création d’un tableau HTML.  
- [ ] **CE3** Le candidat a analysé la stabilité du modèle dans le temps et défini des actions correctives.  
    - Analyse du tableau `evidently`, conclusion sur un éventuel drift.

