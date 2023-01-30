"""
AI-accelerated surrogate model for Aqueous Solubility prediction
One day build to aid understanding for an interview - so do not judge the code so harshly!

Data from:
https://www.kaggle.com/datasets/sorkun/aqsoldb-a-curated-aqueous-solubility-dataset

Network architecture inspired by the following papers:
---

Model is a Multi-Layer Perceptron (currently)


"""
import logging
import pandas as pd
import torch

class InvalidParameterError(Exception):
    """
    One of the user-modifiable parameters is incorrect or beyond reasonable range
    """
    def __init__(self,msg):
        super().__init__(msg)



def load_data(csv_path, train_factor=0.8):
    logging.info("loading data")
    df = pd.read_csv(csv_path,header=0)
   
 

   
    # Shuffle the data
    df = df.sample(frac=1).reset_index(drop=True) # drop=True stops new column of old index

    # Removing the 'G' from atomic group so numeric. Could consider converting to valence shell electrons?
    df["Group"] =  df["Group"].str[1:]

    # for i in range(0,30):
    #     print(df.at[i,"Group"]) # iloc for -ve indexing but slower


    # Split into train and test (no validation)
    if train_factor <= 0 or train_factor > 1:
        raise(InvalidParameterError("Test-Train split is negative or larger than 1"))

    split=int(df.shape[0]*train_factor)
    train_df = df.iloc[:split, :]
    test_df = df.iloc[split:, :]

    logging.debug(f'Split {df.shape[0]} into train: {train_df.shape[0]} and test: {test_df.shape[0]}')

    # Seperate out the inputs and Solubility target
    logging.debug("Splitting data and converting to tensor")
    # TODO: This scraps periodic group as it is not numeric. It also scraps occurence as it's before group column and so is convenient
    # We can deal with this later if we have time, then adjust input layer number or make it dynamic.
    # Could easily change group to numeric or perhaps even number of valence shell electrons, which seems more intuitive for the model??
    train_input = train_df.iloc[:, 7:]
    test_input = train_df.iloc[:, 7:]

    train_aqsol = train_df.iloc[:, 5]
    test_aqsol = test_df.iloc[:, 5]


    




def main():
    logging.basicConfig(format='%(levelname)s at %(asctime)s\n%(message)s\n', datefmt='%M:%S', level=logging.DEBUG)
    df = load_data("curated-solubility-dataset.csv",train_factor=0.8)
    logging.info("data loaded")

    # for i in range(0,):
    #     print(df.at[0,"ID"]) # iloc for -ve indexing but slower

    
    




if __name__ == "__main__":
    main()