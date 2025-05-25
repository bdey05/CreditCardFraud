import numpy as np
import pandas as pd 


def split_csv() -> None:
    """
        Creates four different CSV files from the original dataset CSV file:
            1 initial CSV file that will be used for model training/validation/testing
            2 CSV files that will be used as additional training data for the model (i.e. re-trigger model training)
            1 CSV file that will be passed to the final model for predicting 
    """
    #Reads in the original dataset and shuffles the rows in it 
    df = pd.read_csv("./data/original_dataset.csv").sample(frac=1, random_state=4).reset_index(drop=True)

    #Creates 4 separate dataframes from the original dataset
    dfs = np.array_split(df, 4)

    #Check the percentage of fraudulent transactions in each split
    for split in dfs:
        print("Fraud Percentage: ", (split['Class'].value_counts()[1]/split['Class'].value_counts()[0]) * 100)

    #Save each dataframe as a CSV file locally 
    dfs[0].to_csv("./data/initial_dataset.csv", index=False)
    dfs[1].to_csv("./data/training_data/dataset1.csv", index=False)
    dfs[2].to_csv("./data/training_data/dataset2.csv", index=False)
    dfs[3].to_csv("./data/prediction_dataset.csv", index=False)

if __name__ == "__main__":
    split_csv()