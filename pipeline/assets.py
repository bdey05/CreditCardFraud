import dagster as dg 
import pandas as pd 

@dg.asset
def credit_card_trans() -> pd.DataFrame:
    df = pd.read_csv("../data/initial_dataset.csv")
    return df  

@dg.asset 
def training_test_data() -> tuple: 
    return 

@dg.asset 
def transformed_train_data() -> tuple: 
    return 


@dg.asset 
def transformed_test_data() -> tuple:
    return 

@dg.asset 
def trained_model() -> None:
    return 

