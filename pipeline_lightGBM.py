from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer

import pandas as pd

# Chargement des données sans cible
df = pd.read_csv("application_train.csv").drop(columns=["TARGET"], errors="ignore")

# Séparation par type de colonne
num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

# Pipelines
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

# Prétraitement
preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

# Pipeline complet (prétraitement + réduction de dimension + clustering)
pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("pca", PCA(n_components=5)),
    ("clustering", KMeans(n_clusters=3, random_state=42))
])

# Exécution
pipeline.fit(df)

# Résultats : les labels de cluster attribués
cluster_labels = pipeline.named_steps["clustering"].labels_
