"""
AI-accelerated surrogate model for Aqueous Solubility prediction
This was done in around one day only as preparation for an interview - so do not judge the code so harshly!

Data from:
https://www.kaggle.com/datasets/sorkun/aqsoldb-a-curated-aqueous-solubility-dataset

Network architecture inspired by the following papers:


"""
import pandas as pd
import math

class InvalidParameterError(Exception):
    """
    One of the user-modifiable parameters is incorrect or beyond reasonable range
    """
    def __init__(self,msg):
        super().__init__(msg)



def load_data(csv_path, train_factor=0.8):
    print("loading data")
    df = pd.read_csv(csv_path,header=0)
   
    # Shuffle the data
    df = df.sample(frac=1).reset_index(drop=True) # drop=True stops new column of old index
    

    # Split into train and test (no validation)
    if train_factor <= 0 or train_factor > 1:
        raise(InvalidParameterError("Test-Train split is negative or larger than 1"))

    




def main():
    df = load_data("curated-solubility-dataset.csv",train_factor=0.8)
    print("data loaded")

    for i in range(0,):
        print(df.at[0,"ID"]) # iloc for -ve indexing but slower

    
    




if __name__ == "__main__":
    main()