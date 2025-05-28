import dagster as dg
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier 
from sklearn.model_selection import RandomizedSearchCV
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
def best_candidate_model(training_data: tuple[pd.DataFrame, pd.Series]) -> dg.Output:
    """
    The best performing model asset between 2 candidate models
    """

    X_train, y_train = training_data

    #Calculate the weight for the imbalanced dataset to be used by XGBoost
    weight = y_train.value_counts()[0] / y_train.value_counts()[1]

    #Create an XGBoost pipeline that standardizes the only 2 features not a result of PCA (Amount and Time)
    pipe_xgb = make_pipeline(
        ColumnTransformer(
            [("robust_scaler", RobustScaler(), ["Amount", "Time"])],
            remainder="passthrough",
        ),
        XGBClassifier(eval_metric="aucpr", scale_pos_weight=weight),
    )

    #Define XGBoost param grid for Randomized Search 
    xgb_param_grid = {
        "xgbclassifier__n_estimators": [200, 250, 300],  
        "xgbclassifier__learning_rate": [0.01, 0.1, 0.2], 
        "xgbclassifier__max_depth": [3, 5, 7],          
    }

    #Create a Random Forest pipeline that standardizes the only 2 features not a result of PCA (Amount and Time)
    pipe_rfc = make_pipeline(
        ColumnTransformer(
            [("robust_scaler", RobustScaler(), ["Amount", "Time"])],
            remainder="passthrough",
        ),
        RandomForestClassifier(max_features="sqrt", class_weight="balanced"),
    )

    #Define Random Forest param grid for Randomized Search 
    rfc_param_grid = {
        "randomforestclassifier__n_estimators": [200, 250, 300],
        "randomforestclassifier__max_depth": [None, 10, 20, 30],
    }

    #Create list of candidate models and temporary variables to hold the best model found from Randomized Search along with its best score, params, and model type
    candidate_models = [
        ("XGBoost", pipe_xgb, xgb_param_grid),
        ("Random Forest", pipe_rfc, rfc_param_grid),
    ]
    best_score = -50.0
    best_model = None
    best_model_type = ""
    best_params = {}

    #Run Randomized Search for each candidate model using average precision as the scoring metric
    for candidate_model in candidate_models:
        model_type, pipe_estimator, est_param_grid = candidate_model
        rs = RandomizedSearchCV(
            estimator=pipe_estimator,
            param_distributions=est_param_grid,
            n_iter=10,
            n_jobs=-1,
            cv=5,
            scoring="average_precision"
        )
        #Fit the Randomized Search on the training data and replace the best model if the current model returns a higher score
        rs.fit(X_train, y_train)
        if rs.best_score_ > best_score:
            best_score = rs.best_score_
            best_model = pipe_estimator
            best_params = rs.best_params_
            best_model_type = model_type

    #Stores the best score from Randomized Search as metadata
    return dg.Output(
        (best_model, best_params, best_model_type),
        metadata={
            "best_rs_score": float(best_score),
        },
    )


@dg.asset
def final_trained_model(
    best_candidate_model: dg.Output,
    training_data: tuple[pd.DataFrame, pd.Series],
    test_data: tuple[pd.DataFrame, pd.Series],
) -> dg.Output:
    """
    The best performing model asset that is fitted on the entirety of the training data
    """
    best_model, best_params, best_model_type = best_candidate_model
    X_train, y_train = training_data
    X_test, y_test = test_data

    #Fit the best model with its best params on the training data
    final_model = best_model 
    final_model.set_params(**best_params)
    final_model.fit(X_train, y_train)

    #Save the best model as a joblib file for the FLask API to access
    model_path = "../model/final_model.joblib"
    joblib.dump(final_model, model_path)

    #Calculate final average precision score on the test dataset
    y_score = final_model.predict_proba(X_test)[:, 1]
    final_score = average_precision_score(y_true=y_test, y_score=y_score)

    return dg.Output(
        model_path,
        metadata={
            "final_model_type": best_model_type,
            "final_score": float(final_score),
            "final_params": best_params,
        },
    )
