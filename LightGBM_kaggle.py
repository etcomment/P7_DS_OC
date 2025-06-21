# HOME CREDIT DEFAULT RISK COMPETITION
# Most features are created by applying min, max, mean, sum and var functions to grouped tables.
# Little feature selection is done and overfitting might be a problem since many features are related.
# The following key ideas were used:
# - Divide or subtract important features to get rates (like annuity and income)
# - In Bureau Data: create specific features for Active credits and Closed credits
# - In Previous Applications: create specific features for Approved and Refused applications
# - Modularity: one function for each table (except bureau_balance and application_test)
# - One-hot encoding for categorical features
# All tables are joined with the application DF using the SK_ID_CURR key (except bureau_balance).
# You can use LightGBM with KFold or Stratified KFold.
import sys

# Update 16/06/2018:
# - Added Payment Rate feature
# - Removed index from features
# - Use standard KFold CV (not stratified)

#(base) stiven@stiven-fixe:~/Documents/formation data scientist/stiven/source/P7_DS_OC$ mlflow ui --backend-store-uri file:/home/stiven/Documents/formation\ data\ scientist/stiven/source/P7_DS_OC/mlruns


import numpy as np
import pandas as pd
import gc
import time
from contextlib import contextmanager
from lightgbm import LGBMClassifier
import lightgbm as lgb
from scipy.stats import boxcox, yeojohnson
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, RidgeClassifier
from sklearn.metrics import roc_auc_score, roc_curve, recall_score, precision_score, f1_score, mean_squared_error, \
    r2_score, mean_absolute_error, make_scorer
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import mlflow
from sklearn.dummy import DummyClassifier
import mlflow.data
from mlflow.data.pandas_dataset import PandasDataset
from sklearn.metrics import confusion_matrix
import mlflow.lightgbm
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import shap
from sklearn.preprocessing import OrdinalEncoder
from evidently import Report
from evidently import Dataset, DataDefinition
from evidently.descriptors import Sentiment, TextLength, Contains
from evidently.presets import TextEvals
from evidently.presets import DataDriftPreset
import os

warnings.simplefilter(action='ignore', category=FutureWarning)

import re

def clean_column_names(columns):
    cleaned = []
    for col in columns:
        # Remplacer tout ce qui n'est pas lettre, chiffre ou underscore par un underscore
        new_col = re.sub(r'[^A-Za-z0-9_]', '_', col)
        cleaned.append(new_col)
    return cleaned

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))


# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category=True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category, dtype=float)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


# Preprocess application_train.csv and application_test.csv
def application_train_test(num_rows=None, nan_as_category=False):
    # Read data and merge
    df = pd.read_csv('./artefacts/data/application_train.csv', nrows=num_rows, encoding='ISO-8859-1')
    test_df = pd.read_csv('./artefacts/data/application_test.csv', nrows=num_rows, encoding='ISO-8859-1')
    #print("Train samples: {}, test samples: {}".format(len(df), len(test_df)))
    print("Train samples: {}".format(len(df)))
    df = pd.concat([df,test_df], ignore_index=True).reset_index()
    # Optional: Remove 4 applications with XNA CODE_GENDER (train set)
    df = df[df['CODE_GENDER'] != 'XNA']

    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])
    # Categorical features with One-Hot encode
    df, cat_cols = one_hot_encoder(df, nan_as_category)

    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    # Some simple new features (percentages)
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    #del test_df
    gc.collect()
    return df


# Preprocess bureau.csv and bureau_balance.csv
def bureau_and_balance(num_rows=None, nan_as_category=True):
    bureau = pd.read_csv('./artefacts/data/bureau.csv', nrows=num_rows, encoding='ISO-8859-1')
    bb = pd.read_csv('./artefacts/data/bureau_balance.csv', nrows=num_rows, encoding='ISO-8859-1')
    bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)

    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace=True)
    del bb, bb_agg
    gc.collect()

    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }
    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat: cat_aggregations[cat] = ['mean']
    for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']

    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    gc.collect()
    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    del closed, closed_agg, bureau
    gc.collect()
    return bureau_agg


# Preprocess previous_applications.csv
def previous_applications(num_rows=None, nan_as_category=True):
    prev = pd.read_csv('./artefacts/data/previous_application.csv', nrows=num_rows, encoding='ISO-8859-1')
    prev, cat_cols = one_hot_encoder(prev, nan_as_category=True)
    # Days 365.243 values -> nan
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)
    # Add feature: value ask / value received percentage
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']

    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    del refused, refused_agg, approved, approved_agg, prev
    gc.collect()
    return prev_agg


# Preprocess POS_CASH_balance.csv
def pos_cash(num_rows=None, nan_as_category=True):
    pos = pd.read_csv('./artefacts/data/POS_CASH_balance.csv', nrows=num_rows, encoding='ISO-8859-1')
    pos, cat_cols = one_hot_encoder(pos, nan_as_category=True)
    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']

    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    del pos
    gc.collect()
    return pos_agg


# Preprocess installments_payments.csv
def installments_payments(num_rows=None, nan_as_category=True):
    ins = pd.read_csv('./artefacts/data/installments_payments.csv', nrows=num_rows, encoding='ISO-8859-1')
    ins, cat_cols = one_hot_encoder(ins, nan_as_category=True)
    # Percentage and difference paid in each installment (amount paid and installment value)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    del ins
    gc.collect()
    return ins_agg


# Preprocess credit_card_balance.csv
def credit_card_balance(num_rows=None, nan_as_category=True):
    cc = pd.read_csv('./artefacts/data/credit_card_balance.csv', nrows=num_rows, encoding='ISO-8859-1')
    cc, cat_cols = one_hot_encoder(cc, nan_as_category=True)
    # General aggregations
    cc.drop(['SK_ID_PREV'], axis=1, inplace=True)
    cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    del cc
    gc.collect()
    return cc_agg

def do_drift_report():

    report = Report([
        DataDriftPreset(method="psi")
    ],
        include_tests="True")
    # rapport a faire entre application train et application test
    my_eval = report.run(pd.read_csv("train.csv"), pd.read_csv("test.csv"))
    my_eval.save_html("rapport.html")

def split_and_impute(df, impute=True):
    # Nettoyage des valeurs infinies
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_test = (df[df['TARGET'].isna()].drop(columns=['TARGET']))
    df_test.to_csv("test.csv", index=False)
    df = df[df["TARGET"].notna()].copy()
    #on split dans un val pour pouvoir avoir un jeu de donnée dont on connais les target, et qu'on va pouvoir utiliser pour valider le modele
    df, df_val = train_test_split(df,test_size=0.01, random_state=42, stratify=df['TARGET'])
    df_val.to_csv("val.csv")

    df_imputed = pd.DataFrame(df, columns=df.columns)
    df_imputed.to_csv("train.csv")

    if (impute) :
        print("Imutation des données")
        imputer = SimpleImputer(strategy='most_frequent')
        df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
        colonnes_onehot = [col for col in df_imputed.columns
                           if df_imputed[col].nunique() == 2 and set(df_imputed[col].unique()).issubset({0, 1})]
        for i in df.columns[4:]:
            if i not in colonnes_onehot:
                skewness = df[i].skew()
                if (skewness < -50) or (skewness > 50):
                    try :
                        print("Skew " + str(i) + " :" + str(skewness))
                        transformed, _ = yeojohnson(df[i])
                        df.loc[:, i] = transformed.astype(df[i].dtype)
                    except ValueError as e:
                        print(e)
        df_imputed.to_csv("train_imputed.csv", index=False)

    X_train, X_test, y_train, y_test = train_test_split(
        df_imputed.drop(['TARGET'],axis=1), df_imputed['TARGET'],
        test_size=0.20, random_state=42, stratify=df_imputed['TARGET'])
    print("Fin d'imputation : Train shape: {}, test shape: {}".format(X_train.shape, X_test.shape))

    return X_train, y_train, X_test, y_test

def business_cost_score(y_true, y_pred_proba, fn_cost=10, fp_cost=1, threshold=0.5):
    y_pred = (y_pred_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return -(fn * fn_cost + fp * fp_cost)  # Négatif car GridSearchCV maximise

def metric_cout_metier(y_pred, dataset):
    y_true = dataset.get_label()
    seuil = 0.5  # seuil fixe ici
    y_pred_bin = (y_pred >= seuil).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_bin).ravel()
    cost = fn * 10 + fp * 1
    return "business_cost", cost, False  # False = on cherche à MINIMISER

def resume_modeles(modele, auc, cout):
    file_path = "resume_modeles.csv"
    nouvelle_ligne = pd.DataFrame([{
        'model_name': modele,
        'auc_valid': auc,
        'business_cost': cout
    }])

    # Si le fichier existe, on l’ouvre et on ajoute
    if os.path.exists(file_path):
        df_resultats = pd.read_csv(file_path)
        df_resultats = pd.concat([df_resultats, nouvelle_ligne], ignore_index=True)
    else:
        # Sinon, on crée un nouveau DataFrame
        df_resultats = nouvelle_ligne

    # Sauvegarde
    df_resultats.to_csv(file_path, index=False)
    print(f"✅ Résultat ajouté pour {modele}")

def find_best_threshold_business(y_true, y_pred_proba, fn_cost=10, fp_cost=1):
    best_cost = float('inf')
    best_thresh = 0.5
    for thresh in np.linspace(0.01, 0.99, 99):
        y_pred_bin = (y_pred_proba >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_bin).ravel()
        cost = fn * fn_cost + fp * fp_cost
        if cost < best_cost:
            best_cost = cost
            best_thresh = thresh
    return {"best_threshold": best_thresh, "best_cost": best_cost}

# LightGBM GBDT with KFold or Stratified KFold
# Parameters from Tilii kernel: https://www.kaggle.com/tilii7/olivier-lightgbm-parameters-by-bayesian-opt/code
#def kfold_lightgbm(df, num_folds, stratified=True, debug=False):
def kfold_lightgbm(X_train, y_train, X_test, y_test, num_folds, stratified=True, debug=False):
    # Tentative de conversion des colonnes object en float
    # Liste pour les colonnes catégorielles problématiques
    categorical = []
    for col in X_train.columns:
        if X_train[col].dtype == 'object':
            X_train[col] = X_train[col].astype('category')
            X_test[col] = X_test[col].astype('category')
            categorical.append(col)

    # Log des paramètres d'expérience
    mlflow.log_param("n_folds", num_folds)

    if debug:
        X_train = X_train.iloc[:10000].copy()
        X_test = X_test.iloc[:10000].copy()
        y_train = y_train.iloc[:10000].copy()
        print("🔧 Mode debug activé : jeu d'entraînement réduit à 10 000 lignes")

    # Divide in training/validation and test data
    train_df = X_train
    test_df = X_test
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    print("#### STARTING lightGBM  ####")
    #del df
    gc.collect()
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=1001)
    else:
        folds = KFold(n_splits=num_folds, shuffle=True, random_state=1001)
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]

    # Nettoyage des noms de colonnes
    clean_feats = clean_column_names(feats)
    rename_dict = dict(zip(feats, clean_feats))

    train_df.rename(columns=rename_dict, inplace=True)
    test_df.rename(columns=rename_dict, inplace=True)
    feats = clean_feats

    # Log de la liste des features dans MLflow
    mlflow.log_param("feature_count", len(feats))
    mlflow.log_dict({"features": feats}, "features.json")
    with open("features_used.txt", "w") as f:
        f.write("\n".join(feats))

    mlflow.log_artifact("features_used.txt")

    # Nettoyage des noms de colonnes catégorielles également
    categorical_clean = [rename_dict[c] for c in categorical if c in rename_dict]

    # Supposons que train_df est ton DataFrame d'entraînement
    feature_types = train_df.dtypes.reset_index()
    feature_types.columns = ['feature', 'dtype']

    # Sauvegarder dans un fichier temporaire
    features_path = "feature_types.csv"
    feature_types.to_csv(features_path, index=False)

    # Enregistrer comme artefact dans MLflow
    mlflow.log_artifact(features_path, artifact_path="metadata")

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], y_train)):
        train_x, train_y = train_df[feats].iloc[train_idx], y_train.iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], y_train.iloc[valid_idx]

        sample_weights = train_y.map({1: 10, 0: 1})

        # Construction des datasets LightGBM avec support des colonnes catégorielles
        lgb_train = lgb.Dataset(train_x, label=train_y, weight=sample_weights, categorical_feature=categorical_clean, free_raw_data=False)
        lgb_valid = lgb.Dataset(valid_x, label=valid_y, categorical_feature=categorical_clean, reference=lgb_train, free_raw_data=False)

        params = {
            'objective': 'binary',
            'boosting_type': 'gbdt',
            'learning_rate': 0.02,
            'num_leaves': 34,
            'colsample_bytree': 0.9497036,
            'subsample': 0.8715623,
            'max_depth': 8,
            'reg_alpha': 0.041545473,
            'reg_lambda': 0.0735294,
            'min_split_gain': 0.0222415,
            'min_child_weight': 39.3259775,
            'verbosity': -1,
            'nthread': 8,
            'metric': 'auc',
            'is_unbalanced': False
        }
        print("########TRAINING#######")
        fold_start_time = time.time()

        clf = lgb.train(
            params,
            lgb_train,
            num_boost_round=1000,
            valid_sets=[lgb_train, lgb_valid],
            valid_names=['train', 'valid'],
            feval=metric_cout_metier,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=200)
            ]
        )
        fold_duration = time.time() - fold_start_time
        mlflow.log_metric(f"fold_{n_fold}_duration_sec", fold_duration)

        # Prédictions
        train_pred = clf.predict(train_x, num_iteration=clf.best_iteration)
        valid_pred = clf.predict(valid_x, num_iteration=clf.best_iteration)

        train_score = roc_auc_score(train_y, train_pred)
        valid_score = roc_auc_score(valid_y, valid_pred)

        # Log MLflow
        mlflow.log_metric(f"fold_{n_fold+1}_train_auc", train_score)
        mlflow.log_metric(f"fold_{n_fold+1}_valid_auc", valid_score)
        mlflow.log_params(params)
        mlflow.log_metric(f"train_auc", train_score)
        mlflow.log_metric(f"valid_auc", valid_score)

        result = find_best_threshold_business(valid_y, valid_pred)
        mlflow.log_metric(f"fold_{n_fold + 1}_business_cost_opt", result["best_cost"])
        mlflow.log_metric(f"fold_{n_fold + 1}_business_thresh_opt", result["best_threshold"])
        print(
            f"Fold {n_fold + 1} | Coût métier optimisé = {result['best_cost']} @ seuil = {result['best_threshold']:.2f}")

        oof_preds[valid_idx] = clf.predict(valid_x, num_iteration=clf.best_iteration)
        sub_preds += clf.predict(test_df[feats], num_iteration=clf.best_iteration) / folds.n_splits

        #export des faux negatifs
        # prédiction binaire avec seuil optimal
        preds_binary = (valid_pred >= result["best_threshold"]).astype(int)

        # repérage des faux négatifs (vrai = 1, prédit = 0)
        faux_negatifs_mask = (valid_y == 1) & (preds_binary == 0)

        # extraire les lignes
        faux_negatifs_df = valid_x.loc[faux_negatifs_mask].copy()
        faux_negatifs_df["vrai_label"] = valid_y[faux_negatifs_mask]
        faux_negatifs_df["pred_proba"] = valid_pred[faux_negatifs_mask]

        # sauvegarde dans un fichier CSV
        csv_path = f"faux_negatifs_fold_{n_fold + 1}.csv"
        faux_negatifs_df.to_csv(csv_path, index=False)

        # Log MLflow si tu veux
        mlflow.log_artifact(csv_path, artifact_path="faux_negatifs")

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importance()
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        resume_modeles("LightGBM", valid_score, result['best_cost'])
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(train_df[feats], y_train)
    dummy_preds = dummy.predict_proba(train_df[feats])[:, 1]
    dummy_score = roc_auc_score(y_train, dummy_preds)

    print("Dummy AUC score: {:.6f}".format(dummy_score))
    mlflow.log_metric("dummy_auc", dummy_score)

    print('Full AUC score %.6f' % roc_auc_score(y_train, oof_preds))
    # Write submission file and plot feature importance
    if not debug:
        y_test = sub_preds
        test_df[['SK_ID_CURR']].to_csv(submission_file_name, index=False)
    display_importances(feature_importance_df)

    # Enregistrer l'importance des features comme artefact MLflow
    fi_path = "feature_importances.csv"
    feature_importance_df.to_csv(fi_path, index=False)
    mlflow.log_artifact(fi_path)

    # Logger un modèle entraîné (exemple : le dernier modèle d'entraînement en mémoire, ou un retrain global)
    final_model = lgb.LGBMClassifier(**params)
    final_model.fit(train_df[feats], y_train, categorical_feature=categorical_clean)

    mlflow.lightgbm.log_model(final_model, artifact_path="model")
    #show_shap_summary(final_model,train_df[feats])

    return feature_importance_df

def kfold_lightgbm_gridsearch(X_train, y_train, X_test, y_test, num_folds, stratified=True, debug=False):
    # Tentative de conversion des colonnes object en float
    # Liste pour les colonnes catégorielles problématiques

    print("#### STARTING lightGBM avec GRIDSEARSHCV ####")

    categorical = []
    for col in X_train.columns:
        if X_train[col].dtype == 'object':
            X_train[col] = X_train[col].astype('category')
            X_test[col] = X_test[col].astype('category')
            categorical.append(col)

    # Log des paramètres d'expérience
    mlflow.log_param("n_folds", num_folds)

    if debug:
        X_train = X_train.iloc[:10000].copy()
        X_test = X_test.iloc[:10000].copy()
        y_train = y_train.iloc[:10000].copy()
        y_test = y_test.iloc[:10000].copy()
        print("🔧 Mode debug activé : jeu d'entraînement réduit à 10 000 lignes")

    # Divide in training/validation and test data
    train_df = X_train
    test_df = X_test
    print("Starting LightGBM with GridSearchCV. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    #del df
    gc.collect()

    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]

    # Nettoyage des noms de colonnes
    clean_feats = clean_column_names(feats)
    rename_dict = dict(zip(feats, clean_feats))

    train_df.rename(columns=rename_dict, inplace=True)
    test_df.rename(columns=rename_dict, inplace=True)
    feats = clean_feats

    # Log de la liste des features dans MLflow
    mlflow.log_param("feature_count", len(feats))
    mlflow.log_dict({"features": feats}, "features.json")
    with open("features_used.txt", "w") as f:
        f.write("\n".join(feats))

    mlflow.log_artifact("features_used.txt")

    # Nettoyage des noms de colonnes catégorielles également
    categorical_clean = [rename_dict[c] for c in categorical if c in rename_dict]

    # Supposons que train_df est ton DataFrame d'entraînement
    feature_types = train_df.dtypes.reset_index()
    feature_types.columns = ['feature', 'dtype']

    # Sauvegarder dans un fichier temporaire
    features_path = "feature_types.csv"
    feature_types.to_csv(features_path, index=False)

    # Enregistrer comme artefact dans MLflow
    mlflow.log_artifact(features_path, artifact_path="metadata")

    # Définir les paramètres de base
    params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'metric': 'auc',
        'is_unbalanced': True,
        'verbosity': -1,
        'subsample_for_bin': 100000,
        'nthread': 4
    }

    param_grid = {
        'learning_rate': [0.01, 0.02],  # Ajout d'une troisième valeur pour le taux d'apprentissage
        'num_leaves': [20, 30],  # Ajout d'une troisième valeur pour le nombre de feuilles
        'max_depth': [5],  # Ajout d'une troisième valeur pour la profondeur maximale
        'reg_alpha': [0.04],  # Une seule valeur pour la régularisation alpha
        'reg_lambda': [0.07],  # Une seule valeur pour la régularisation lambda
    }

    # Créer un modèle LightGBM compatible avec scikit-learn
    lgb_model = lgb.LGBMClassifier(**params)

    business_scorer = make_scorer(business_cost_score, needs_proba=True)

    # Initialiser GridSearchCV
    grid_search = GridSearchCV(
        estimator=lgb_model,
        param_grid=param_grid,
        scoring=business_scorer,
        cv=StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42),
        verbose=0,
        n_jobs=4
    )

    smote = SMOTE(sampling_strategy=0.5,random_state=42)
    train_x, train_y = smote.fit_resample(train_df, y_train)
    print("POST SMOTE Train shape: {}, test shape: {}".format(X_train.shape, y_train.shape))

    # Exécuter GridSearchCV
    grid_search.fit(train_x, train_y)

    # Afficher les meilleurs paramètres et le meilleur score
    print("Meilleurs paramètres trouvés: ", grid_search.best_params_)
    print("Meilleur score (coût métier): ", -grid_search.best_score_)


    # Utiliser le meilleur modèle pour faire des prédictions
    best_clf = grid_search.best_estimator_
    train_pred = best_clf.predict_proba(train_x)[:, 1]
    valid_pred = best_clf.predict_proba(X_test)[:, 1]

    # Calculer les scores
    train_score = roc_auc_score(train_y, train_pred)
    valid_score = roc_auc_score(y_test, valid_pred)

    print('Train AUC : %.6f' % train_score)
    print('Valid AUC : %.6f' % valid_score)

    # Log MLflow
    mlflow.log_metric("train_auc", train_score)
    mlflow.log_metric("valid_auc", valid_score)
    mlflow.log_metric("business_cost_score", -grid_search.best_score_)
    mlflow.log_params(grid_search.best_params_)

    # Nettoyage
    del best_clf, train_x, train_y
    gc.collect()

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(train_df[feats], y_train)
    dummy_preds = dummy.predict_proba(train_df[feats])[:, 1]
    dummy_score = roc_auc_score(y_train, dummy_preds)

    print("Dummy AUC score: {:.6f}".format(dummy_score))
    mlflow.log_metric("dummy_auc", dummy_score)

    # Write submission file and plot feature importance
    if not debug:
        y_test = sub_preds
        test_df[['SK_ID_CURR']].to_csv(submission_file_name, index=False)

    # Enregistrer l'importance des features comme artefact MLflow
    fi_path = "feature_importances.csv"
    feature_importance_df.to_csv(fi_path, index=False)
    mlflow.log_artifact(fi_path)

    # Logger un modèle entraîné (exemple : le dernier modèle d'entraînement en mémoire, ou un retrain global)
    final_model = lgb.LGBMClassifier(**params)
    final_model.fit(train_df[feats], y_train, categorical_feature=categorical_clean)

    # Récupérer les importances des caractéristiques
    importances = final_model.feature_importances_

    # Créer le DataFrame
    feature_importance_df = pd.DataFrame({
        'feature': feats,
        'importance': importances
    })

    display_importances(feature_importance_df)

    mlflow.lightgbm.log_model(final_model, artifact_path="model")
    #show_shap_summary(final_model,train_df[feats])
    print("#### FIN lightGBM avec GRIDSEARSHCV ####")
    resume_modeles("LightGBM avec grid", valid_score, -grid_search.best_score_)
    return feature_importance_df

# Display/plot feature importance
def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance",
                                                                                                   ascending=False)[
           :40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    #plt.figure(figsize=(8, 10))
    #sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    #plt.title('LightGBM Features (avg over folds)')
    #plt.tight_layout()
    #plt.savefig('lgbm_importances01.png')

def show_shap_summary(model, X, max_display=20, plot_type="bar"):
    """
    Affiche un graphe SHAP summary pour un modèle LightGBM ou LGBMClassifier
    :param model: modèle entraîné (lgb.Booster ou LGBMClassifier)
    :param X: DataFrame utilisé pour le calcul des SHAP values
    :param max_display: nombre de features à afficher
    :param plot_type: "bar" ou "dot"
    """
    # Vérification : convertir en Booster si nécessaire
    if hasattr(model, "booster"):
        booster = model.booster_
    else:
        booster = model

    # Initialisation de l'explainer
    explainer = shap.TreeExplainer(booster)

    # Calcul des valeurs SHAP
    shap_values = explainer.shap_values(X)

    # Affichage summary plot
    #plt.figure(figsize=(12, 6))
    #shap.summary_plot(shap_values, X, max_display=max_display, plot_type=plot_type)
    #plt.savefig('lgbm_shap_global.png')
    show_shap_for_single_prediction(model, X)

def show_shap_for_single_prediction(model, X, row_index=1):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    #plt.savefig('lgbm_shap_single.png')
    shap.initjs()
    return shap.force_plot(
        explainer.expected_value,
        shap_values[row_index, :],
        X.iloc[row_index, :]
    )

def kfold_ridge_classification(X_train, y_train, X_test, y_test, num_folds=5, debug=False):

    print("#### STARTING Ridge Classification ####")
    categorical = [col for col in X_train.columns if X_train[col].dtype == 'object' or str(X_train[col].dtype).startswith('category')]
    mlflow.log_param("n_folds", num_folds)

    if debug:
        X_train = X_train.iloc[:10000].copy()
        X_test = X_test.iloc[:10000].copy()
        y_train = y_train.iloc[:10000].copy()
        print("\U0001F527 Mode debug activé : 10 000 lignes")

    train_df = X_train.copy()
    test_df = X_test.copy()


    print("Starting Ridge Classification. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    gc.collect()

    folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['TARGET', 'SK_ID_CURR', 'index']]

    clean_feats = [f.replace(" ", "_").replace("(", "").replace(")", "") for f in feats]
    rename_dict = dict(zip(feats, clean_feats))
    train_df.rename(columns=rename_dict, inplace=True)
    test_df.rename(columns=rename_dict, inplace=True)
    feats = clean_feats

    mlflow.log_param("feature_count", len(feats))
    mlflow.log_dict({"features": feats}, "features.json")
    with open("features_used.txt", "w") as f:
        f.write("\n".join(feats))
    mlflow.log_artifact("features_used.txt")

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], y_train)):
        train_x = train_df[feats].iloc[train_idx].copy()
        valid_x = train_df[feats].iloc[valid_idx].copy()
        train_y = y_train.iloc[train_idx]
        valid_y = y_train.iloc[valid_idx]

        model = RidgeClassifier()
        model.fit(train_x, train_y)

        valid_pred = model.decision_function(valid_x)
        oof_preds[valid_idx] = valid_pred
        sub_preds += model.decision_function(test_df[feats]) / folds.n_splits

        auc = roc_auc_score(valid_y, valid_pred)
        mlflow.log_metric(f"fold_{n_fold + 1}_auc", auc)

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = model.coef_.flatten()
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        print(f"Fold {n_fold + 1} | AUC: {auc:.4f}")
        gc.collect()

    overall_auc = roc_auc_score(y_train, oof_preds)
    print(f"\u2705 AUC global : {overall_auc:.4f}")
    mlflow.log_metric("overall_auc", overall_auc)

    fi_path = "feature_importances_ridge.csv"
    feature_importance_df.to_csv(fi_path, index=False)
    mlflow.log_artifact(fi_path)

    display_importances(feature_importance_df)

    # Coût global
    business_global = find_best_threshold_business(y_train, oof_preds)
    print(f"✅ Coût métier global : {business_global['best_cost']} @ seuil {business_global['best_threshold']:.2f}")
    mlflow.log_metric("overall_business_cost", business_global['best_cost'])
    mlflow.log_metric("overall_best_threshold", business_global['best_threshold'])

    # Enregistrement du modèle final
    mlflow.sklearn.log_model(model, artifact_path="ridge_classifier_model")
    resume_modeles("Ridge Classification", overall_auc, business_global['best_cost'])
    return feature_importance_df

def kfold_random_forest(X_train, y_train, X_test, y_test, num_folds=5, stratified=True, debug=False):

    mlflow.log_param("n_folds", num_folds)
    mlflow.log_param("model", "RandomForestClassifier")
    print("#### STARTING Random Forests ####")
    # Conversion des colonnes catégorielles
    categorical = [col for col in X_train.columns if X_train[col].dtype == 'object' or str(X_train[col].dtype).startswith('category')]

    if debug:
        X_train = X_train.iloc[:10000].copy()
        X_test = X_test.iloc[:10000].copy()
        y_train = y_train.iloc[:10000].copy()
        print("🔧 Mode debug activé : jeu d'entraînement réduit à 10 000 lignes")


    train_df = X_train.copy()
    test_df = X_test.copy()
    print("Starting RandomForest. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    gc.collect()


    folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42) if stratified else KFold(n_splits=num_folds, shuffle=True, random_state=42)

    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()

    feats = [f for f in train_df.columns if f not in ['TARGET', 'SK_ID_CURR', 'index']]

    clean_feats = [f.replace(" ", "_").replace("(", "").replace(")", "") for f in feats]
    rename_dict = dict(zip(feats, clean_feats))
    train_df.rename(columns=rename_dict, inplace=True)
    test_df.rename(columns=rename_dict, inplace=True)
    feats = clean_feats

    mlflow.log_param("feature_count", len(feats))
    mlflow.log_dict({"features": feats}, "features.json")
    with open("features_used_rf.txt", "w") as f:
        f.write("\n".join(feats))
    mlflow.log_artifact("features_used_rf.txt")

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], y_train)):
        train_x = train_df[feats].iloc[train_idx].copy()
        valid_x = train_df[feats].iloc[valid_idx].copy()
        train_y = y_train.iloc[train_idx]
        valid_y = y_train.iloc[valid_idx]

        # SMOTE
        smote = SMOTE(sampling_strategy=0.5,random_state=42)
        train_x, train_y = smote.fit_resample(train_x, train_y)

        # Modèle RandomForest
        params = {
            'n_estimators': 200,
            'max_depth': 10,
            'random_state': 42,
            'n_jobs': -1
        }
        mlflow.log_params(params)

        model = RandomForestClassifier(**params)
        model.fit(train_x, train_y)

        # Prédictions
        valid_pred_proba = model.predict_proba(valid_x)[:, 1]
        oof_preds[valid_idx] = valid_pred_proba
        sub_preds += model.predict_proba(test_df[feats])[:, 1] / folds.n_splits

        # Scores
        auc = roc_auc_score(valid_y, valid_pred_proba)

        # Ajout de la métrique métier
        business = find_best_threshold_business(valid_y, valid_pred_proba)
        mlflow.log_metric(f"fold_{n_fold + 1}_business_cost", business['best_cost'])
        mlflow.log_metric(f"fold_{n_fold + 1}_threshold", business['best_threshold'])

        print(
            f"Fold {n_fold + 1} | AUC: {auc:.4f} | Coût métier : {business['best_cost']} @ seuil {business['best_threshold']:.2f}")
        mlflow.log_metric(f"fold_{n_fold+1}_auc", auc)

        # Feature importance
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = model.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        gc.collect()

    # Score global
    overall_auc = roc_auc_score(y_train, oof_preds)
    print(f"✅ AUC global : {overall_auc:.4f}")
    mlflow.log_metric("overall_auc", overall_auc)
    # Coût global
    business_global = find_best_threshold_business(y_train, oof_preds)
    print(f"✅ Coût métier global : {business_global['best_cost']} @ seuil {business_global['best_threshold']:.2f}")
    mlflow.log_metric("overall_business_cost", business_global['best_cost'])
    mlflow.log_metric("overall_best_threshold", business_global['best_threshold'])

    # Sauvegarde
    fi_path = "feature_importances_rf.csv"
    feature_importance_df.to_csv(fi_path, index=False)
    mlflow.log_artifact(fi_path)
    resume_modeles("random forest ", overall_auc, business_global['best_cost'])

    return feature_importance_df

def main(debug=False):
    num_rows = 10000 if debug else None
    df = application_train_test(num_rows)
    with timer("Process bureau and bureau_balance"):
        bureau = bureau_and_balance(num_rows)
        print("Bureau df shape:", bureau.shape)
        df = df.join(bureau, how='left', on='SK_ID_CURR')
        del bureau
        gc.collect()
    with timer("Process previous_applications"):
        prev = previous_applications(num_rows)
        print("Previous applications df shape:", prev.shape)
        df = df.join(prev, how='left', on='SK_ID_CURR')
        del prev
        gc.collect()
    with timer("Process POS-CASH balance"):
        pos = pos_cash(num_rows)
        print("Pos-cash balance df shape:", pos.shape)
        df = df.join(pos, how='left', on='SK_ID_CURR')
        del pos
        gc.collect()
    with timer("Process installments payments"):
        ins = installments_payments(num_rows)
        print("Installments payments df shape:", ins.shape)
        df = df.join(ins, how='left', on='SK_ID_CURR')
        del ins
        gc.collect()
    with timer("Process credit card balance"):
        cc = credit_card_balance(num_rows)
        print("Credit card balance df shape:", cc.shape)
        df = df.join(cc, how='left', on='SK_ID_CURR')
        del cc
        gc.collect()

    with timer("Run LightGBM with kfold"):
        with mlflow.start_run():
            print("#### LightGBM ####")
            Xtr,yTr,Xtst,Ytst = split_and_impute(df, impute=False)
            feat_importance = kfold_lightgbm(Xtr,yTr,Xtst,Ytst, num_folds=5, stratified=True, debug=False)
    with timer("Run LightGBM with kfold and GRIDSEARCH"):
        with mlflow.start_run():
            print("#### LightGBM avec GridSearchCV ####")
            Xtr,yTr,Xtst,Ytst = split_and_impute(df)
            feat_importance = kfold_lightgbm_gridsearch(Xtr,yTr,Xtst,Ytst, num_folds=3, stratified=True, debug=False)

    with timer("Run ridge classification with kfold"):
        with mlflow.start_run():
            print("#### Ridge ####")
            feat_importance = kfold_ridge_classification(Xtr,yTr,Xtst,Ytst, num_folds=5, debug=False)

    with timer("Run Random Forest with kfold"):
        with mlflow.start_run():
            print("#### Random Forest ####")
            feat_importance = kfold_random_forest(Xtr,yTr,Xtst,Ytst, num_folds=5, stratified=True, debug=False)
    mlflow.log_artifact("resume_modeles.csv")
if __name__ == "__main__":
    submission_file_name = "submission_kernel02.csv"
    #with timer("Full model run"):
    mlflow.set_tracking_uri('http://localhost:5000')
    mlflow.lightgbm.autolog()
    main()
