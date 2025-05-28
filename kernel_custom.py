import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns


# Import des données
df_bureau = pd.read_csv('./artefacts/data/bureau.csv',encoding='ISO-8859-1')
df_bureau_balance = pd.read_csv('./artefacts/data/bureau_balance.csv',encoding='ISO-8859-1')
df_credit_card = pd.read_csv('./artefacts/data/credit_card_balance.csv',encoding='ISO-8859-1')
df_home_credit = pd.read_csv('./artefacts/data/HomeCredit_columns_description.csv',encoding='ISO-8859-1')
df_installments = pd.read_csv('./artefacts/data/installments_payments.csv',encoding='ISO-8859-1')
df_pos_cash = pd.read_csv('./artefacts/data/POS_CASH_balance.csv',encoding='ISO-8859-1')

list_df=[df_bureau,df_bureau_balance,df_pos_cash,df_installments,df_credit_card,df_home_credit]

for df in enumerate(list_df) :
    print(df[1].head(5))


