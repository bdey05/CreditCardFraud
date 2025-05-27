import dagster as dg
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score


@dg.asset
def credit_card_trans() -> pd.DataFrame:
    """
    The DataFrame asset that is created from the initial dataset
    """
    # Load the local CSV as a DataFrame
    df = pd.read_csv("../data/initial_dataset.csv")
    return df


@dg.multi_asset(outs={"training_data": dg.AssetOut(), "test_data": dg.AssetOut()})
def training_test_data(
    credit_card_trans: pd.DataFrame,
) -> tuple[tuple[pd.DataFrame, pd.Series], tuple[pd.DataFrame, pd.Series]]:
    """
    The training and test dataset assets that are created from splitting the initial DataFrame
    """
    # Separate the features and labels from the dataset
    X = credit_card_trans.loc[:, credit_card_trans.columns != "Class"]
    y = credit_card_trans["Class"]

    # Separate the dataset into train and test splits using a 70/30 split while ensuring stratification since the dataset is imbalanced
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

    return (X_train, y_train), (X_test, y_test)


@dg.asset
def best_candidate_model(training_data: tuple[pd.DataFrame, pd.Series]):
    """
    The best performing model asset out of all 3 candidate models
    """
    X_train, y_train = training_data

    pipe_lr = make_pipeline(
        ColumnTransformer([("robust_scaler", RobustScaler, ["Amount", "Time"])]),
        LogisticRegression(),
    )

    lr_param_grid = {
        "logisticregression__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        "logisticregression__penalty": ["l1", "l2"],
        "logisticregression__solver": ["liblinear"],
        "logisticregression__class_weight": [None, "balanced"],
    }

    pipe_rfc = make_pipeline(
        ColumnTransformer([("robust_scaler", RobustScaler, ["Amount", "Time"])]),
        RandomForestClassifier(),
    )

    rfc_param_grid = {
        "randomforest__n_estimators": [100, 200, 300],
        "randomforest__max_depth": [None, 10, 20, 30],
        "randomforest__max_features": ["sqrt", "log2"],
        "randomforest__class_weight": [None, "balanced", "balanced_subsample"],
    }

    candidate_models = [(pipe_lr, lr_param_grid), (pipe_rfc, rfc_param_grid)]

    for candidate_model in candidate_models:
        pipe_estimator, est_param_grid = candidate_model 
        gs = GridSearchCV(
            estimator=pipe_estimator,
            param_grid=est_param_grid,
            scoring="average_precision",
            cv=2
        )
        scores = cross_val_score(gs, X_train, y_train, scoring="average_precision", cv=5)
        
    return


@dg.asset
def final_trained_model(
    training_data: tuple[pd.DataFrame, pd.Series],
    test_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    return


@dg.asset
def model_score() -> None:
    return
