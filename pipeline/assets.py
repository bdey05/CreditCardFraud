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
from sklearn.metrics import average_precision_score
import joblib  


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
def best_candidate_model(training_data: tuple[pd.DataFrame, pd.Series], context: dg.AssetExecutionContext) -> tuple:
    """
    The best performing model asset between 2 candidate models
    """
    X_train, y_train = training_data

    pipe_lr = make_pipeline(
        ColumnTransformer([("robust_scaler", RobustScaler(), ["Amount", "Time"])], remainder="passthrough"),
        LogisticRegression(),
    )

    lr_param_grid = {
        "logisticregression__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        "logisticregression__penalty": ["l1", "l2"],
        "logisticregression__solver": ["liblinear"],
        "logisticregression__class_weight": [None, "balanced"],
    }

    pipe_rfc = make_pipeline(
        ColumnTransformer([("robust_scaler", RobustScaler(), ["Amount", "Time"])], remainder="passthrough"),
        RandomForestClassifier(),
    )

    rfc_param_grid = {
        "randomforestclassifier__n_estimators": [100, 200, 300],
        "randomforestclassifier__max_depth": [None, 10, 20, 30],
        "randomforestclassifier__max_features": ["sqrt", "log2"],
        "randomforestclassifier__class_weight": [None, "balanced", "balanced_subsample"],
    }

    candidate_models = [("Logistic Regression", pipe_lr, lr_param_grid), ("Random Forest", pipe_rfc, rfc_param_grid)]

    best_score = -50.0
    best_model = None 
    best_model_type = ""
    best_params = {}

    for candidate_model in candidate_models:
        model_type, pipe_estimator, est_param_grid = candidate_model 
        gs = GridSearchCV(
            estimator=pipe_estimator,
            param_grid=est_param_grid,
            scoring="average_precision",
            cv=2,
            n_jobs=-1
        )
        scores = cross_val_score(gs, X_train, y_train, scoring="average_precision", cv=5, n_jobs=-1)
        if np.mean(scores) > best_score:
            best_score = np.mean(scores)
            best_model = pipe_estimator
            best_params = est_param_grid
            best_model_type = model_type 

    context.add_output_metadata({
        "best_cv_score": best_score,
    })

    return best_model, best_params, best_model_type


@dg.asset
def final_trained_model(
    best_candidate_model: tuple,
    training_data: tuple[pd.DataFrame, pd.Series],
    test_data: tuple[pd.DataFrame, pd.Series],
    context: dg.AssetExecutionContext
) -> str:
    best_model, best_params, best_model_type = best_candidate_model
    X_train, y_train = training_data
    X_test, y_test = test_data 
    gs = GridSearchCV(
            estimator=best_model,
            param_grid=best_params,
            scoring="average_precision",
            cv=10,
            n_jobs=-1
    )
    gs.fit(X_train, y_train)

    final_params = gs.best_params_ 
    final_model = gs.best_estimator_

    model_path = "../model/final_model.joblib"
    joblib.dump(final_model, model_path)

    y_score = final_model.predict_proba(X_test)[:, 1]
    final_score = average_precision_score(y_true=y_test, y_score=y_score)

    context.add_output_metadata({
        "final_model_type": best_model_type, 
        "final_score": final_score,
        "final_params": final_params 
    })

    return model_path 



