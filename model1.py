
#lightgbm
from sklearn.model_selection import KFold
from lightgbm import early_stopping, log_evaluation
from sklearn.metrics import roc_auc_score, accuracy_score
import lightgbm as lgb
import gc
import mlflow
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

def model(features, test_features, encoding='ohe', n_folds=5):
    """Train and test a light gradient boosting model using
    cross validation.

    Parameters
    --------
        features (pd.DataFrame):
            dataframe of training features to use
            for training a model. Must include the TARGET column.
        test_features (pd.DataFrame):
            dataframe of testing features to use
            for making predictions with the model.
        encoding (str, default = 'ohe'):
            method for encoding categorical variables. Either 'ohe' for one-hot encoding or 'le' for integer label encoding
            n_folds (int, default = 5): number of folds to use for cross validation

    Return
    --------
        submission (pd.DataFrame):
            dataframe with `SK_ID_CURR` and `TARGET` probabilities
            predicted by the model.
        feature_importances (pd.DataFrame):
            dataframe with the feature importances from the model.
        valid_metrics (pd.DataFrame):
            dataframe with training and validation metrics (ROC AUC) for each fold and overall.

    """
    mlflow.set_tracking_uri('http://localhost:5000')
    with mlflow.start_run():

        # Log des paramètres d'expérience
        mlflow.log_param("encoding", encoding)
        mlflow.log_param("n_folds", n_folds)


        # Extract the ids
        train_ids = features['SK_ID_CURR']
        test_ids = test_features['SK_ID_CURR']

        # Extract the labels for training
        labels = features['TARGET']

        # Remove the ids and target
        features = features.drop(columns=['SK_ID_CURR', 'TARGET'])
        test_features = test_features.drop(columns=['SK_ID_CURR'])

        # One Hot Encoding
        if encoding == 'ohe':
            features = pd.get_dummies(features)
            test_features = pd.get_dummies(test_features)
            # Align the dataframes by the columns
            features, test_features = features.align(test_features, join='inner', axis=1)
            # No categorical indices to record
            cat_indices = 'auto'
        # Integer label encoding
        elif encoding == 'le':
            # Create a label encoder
            label_encoder = LabelEncoder()
            # List for storing categorical indices
            cat_indices = []
            # Iterate through each column
            for i, col in enumerate(features):
                if features[col].dtype == 'object':
                    # Map the categorical features to integers
                    features[col] = label_encoder.fit_transform(np.array(features[col].astype(str)).reshape((-1,)))
                    test_features[col] = label_encoder.transform(np.array(test_features[col].astype(str)).reshape((-1,)))
                    # Record the categorical indices
                    cat_indices.append(i)

        # Catch error if label encoding scheme is not valid
        else:
            raise ValueError("Encoding must be either 'ohe' or 'le'")

        print('Training Data Shape: ', features.shape)
        print('Testing Data Shape: ', test_features.shape)

        # Extract feature names
        feature_names = list(features.columns)

        # Convert to np arrays
        features = np.array(features)
        test_features = np.array(test_features)

        # Create the kfold object
        k_fold = KFold(n_splits=n_folds, shuffle=True, random_state=50)

        # Empty array for feature importances
        feature_importance_values = np.zeros(len(feature_names))

        # Empty array for test predictions
        test_predictions = np.zeros(test_features.shape[0])

        # Empty array for out of fold validation predictions
        out_of_fold = np.zeros(features.shape[0])

        # Lists for recording validation and training scores
        valid_scores = []
        train_scores = []

        # Iterate through each fold
        for fold,(train_indices, valid_indices) in enumerate(k_fold.split(features)):
            # Training data for the fold
            train_features, train_labels = features[train_indices], labels[train_indices]
            # Validation data for the fold
            valid_features, valid_labels = features[valid_indices], labels[valid_indices]

            # Create the model
            model = lgb.LGBMClassifier(n_estimators=10000, objective='binary',
                                       class_weight='balanced', learning_rate=0.05,
                                       reg_alpha=0.1, reg_lambda=0.1,
                                       subsample=0.8, n_jobs=-1, random_state=50)

            # Entraînement du modèle avec les callbacks
            model.fit(
                train_features, train_labels,
                eval_metric='auc',
                eval_set=[(valid_features, valid_labels), (train_features, train_labels)],
                eval_names=['valid', 'train'],
                categorical_feature=cat_indices,
                callbacks=[
                    early_stopping(stopping_rounds=100),
                    log_evaluation(period=200)
                ]
            )

            # Record the best iteration
            best_iteration = model.best_iteration_

            # Record the feature importances
            feature_importance_values += model.feature_importances_ / k_fold.n_splits

            # Make predictions
            test_predictions += model.predict_proba(test_features, num_iteration=best_iteration)[:, 1] / k_fold.n_splits

            # Record the out of fold predictions
            out_of_fold[valid_indices] = model.predict_proba(valid_features, num_iteration=best_iteration)[:, 1]

            # Record the best score
            valid_score = model.best_score_['valid']['auc']
            train_score = model.best_score_['train']['auc']

            valid_scores.append(valid_score)
            train_scores.append(train_score)

            # Log des scores individuels par fold
            mlflow.log_metric(f"fold_{fold}_train_auc", train_score)
            mlflow.log_metric(f"fold_{fold}_valid_auc", valid_score)

            # Clean up memory
            gc.enable()
            del model, train_features, valid_features
            gc.collect()

        # Make the submission dataframe
        submission = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': test_predictions})

        # Make the feature importance dataframe
        feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})

        # Overall validation score
        valid_auc = roc_auc_score(labels, out_of_fold)

        # Add the overall scores to the metrics
        valid_scores.append(valid_auc)
        train_scores.append(np.mean(train_scores))

        # Needed for creating dataframe of validation scores
        fold_names = list(range(n_folds))
        fold_names.append('overall')

        # Dataframe of validation scores
        metrics = pd.DataFrame({'fold': fold_names,
                                'train': train_scores,
                                'valid': valid_scores})
        # Log du modèle complet (à la fin du run)
        mlflow.lightgbm.log_model(lgb.LGBMClassifier(), artifact_path="model")
        fi_path = "feature_importances.csv"
        feature_importances.to_csv(fi_path, index=False)
        mlflow.log_artifact(fi_path)

        return submission, feature_importances, metrics



submission, fi, metrics = model(app_train, app_test)
print('Baseline metrics')
print(metrics)

fi_sorted = plot_feature_importances(fi)
#plt.show()
submission.to_csv('baseline_lgb.csv', index = False)

app_train_domain['TARGET'] = train_labels

# Test the domain knolwedge features
submission_domain, fi_domain, metrics_domain = model(app_train_domain, app_test_domain)
print('Baseline with domain knowledge features metrics')
print(metrics_domain)
fi_sorted = plot_feature_importances(fi_domain)
#plt.show()

submission_domain.to_csv('baseline_lgb_domain_features.csv', index = False)