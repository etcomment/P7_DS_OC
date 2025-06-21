import numpy as np
import pandas as pd
import os
from scipy.stats import yeojohnson

df = pd.read_csv("train.csv")

colonnes_onehot = [col for col in df.columns
                   if df[col].nunique() == 2 and set(df[col].unique()).issubset({0, 1})]

print(colonnes_onehot)
for i in df.columns[4:] :
    skewness = df[i].skew()
    if (skewness < -50) or (skewness > 50) :
        print("Skew " + str(i) + " :" + str(skewness))
"""        transformed, _ = yeojohnson(df[i])
        df.loc[:,i] = transformed.astype(df[i].dtype)
        skewness = df[i].skew()
        print("Skew APRES yeojohnson "+str(i)+" :" + str(skewness))"""