"""
AI-accelerated surrogate model for Aqueous Solubility prediction
One day build to aid understanding for an interview - so do not judge the code so harshly!

Data from:
https://www.kaggle.com/datasets/sorkun/aqsoldb-a-curated-aqueous-solubility-dataset

Network architecture inspired by the following papers:
---

Model is a Multi-Layer feedforward (currently)


"""
import logging
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class InvalidParameterError(Exception):
    """
    One of the user-modifiable parameters is incorrect or beyond reasonable range
    """
    def __init__(self,msg):
        super().__init__(msg)


class MLP(torch.nn.Module):

    def __init__(self, input_layers, hidden_layers):
        super(MLP, self).__init__()

        # Very basic, will experiment with more advanced architectures later
        self.fc1 = nn.Linear(input_layers, hidden_layers)
        self.fc2 = nn.Linear(hidden_layers, hidden_layers) # could add another var here
        self.fc3 = nn.Linear(hidden_layers, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        # 1 output so no softmax
        # not binary so no sigmoid
        return x



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
    train_input = train_df.iloc[:, 7:].astype(float)
    test_input = test_df.iloc[:, 7:].astype(float)

    train_aqsol = train_df.iloc[:, 5].astype(float)
    test_aqsol = test_df.iloc[:, 5].astype(float)

    # I'm getting a nan error - code to check data does not introduce it
    # for df in [train_input, test_input, train_aqsol, test_aqsol]:
    #     df = df.to_numpy()
    #     print(np.any(np.isnan(df)))

    train_input = torch.tensor(train_input.values,dtype=torch.float32)
    test_input = torch.tensor(test_input.values,dtype=torch.float32)
    
    train_aqsol = torch.tensor(train_aqsol.values,dtype=torch.float32)
    test_aqsol = torch.tensor(test_aqsol.values,dtype=torch.float32)


    logging.info("data loaded")
    return train_input, test_input, train_aqsol, test_aqsol


def train(model, train_input, train_aqsol, epochs,criterion,optimiser,device):
    model.train()

    

    for epoch in range(epochs):
        logging.info(f"Beginning epoch {epoch+1} of {epochs}")
        optimiser.zero_grad()

        
        # Forward
        output = model(train_input.to(device))
        loss = criterion(output.squeeze(), train_aqsol.to(device).squeeze())
        logging.debug(loss)
        # Backwards
        loss.backward()
        optimiser.step()



def eval(model, test_input,test_aqsol,criterion,device):
    model.eval()
    aqsol_pred = model(test_input.to(device))
    after_train = criterion(aqsol_pred.squeeze(),test_aqsol.to(device).squeeze())
    print(f'test loss {after_train}')
    



def main():
    # Hyperparams
    train_factor = 0.8
    hidden_layers = 64
    epochs = 10
    lr = 0.01



    logging.basicConfig(format='%(levelname)s at %(asctime)s\n%(message)s\n', datefmt='%M:%S', level=logging.DEBUG)

    # TODO: Add list of variables that feeds into load_data() so you can choose inputs to drop
    train_input, test_input, train_aqsol, test_aqsol = load_data("curated-solubility-dataset.csv",train_factor=train_factor)

    input_layers = train_input.shape[1]
    model = MLP(input_layers,hidden_layers)

    # Hybrid laptop problems ugh
    for i in range(torch.cuda.device_count()):
        logging.debug(f'device {i}: {torch.cuda.get_device_name(i)}')
        # device 0: Quadro RTX 3000 with Max-Q Design

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.debug(f'Using device {device}')

    model = model.to(device)
    train_input.to(device)
    train_aqsol.to(device)

    test_input.to(device)
    test_aqsol.to(device)


    logging.debug("Vars sent to device")


    criterion = nn.MSELoss()
    optimiser = optim.SGD(model.parameters(),lr=lr)


    
    train(model, train_input, train_aqsol, epochs, criterion, optimiser,device)
    eval(model, test_input,test_aqsol,criterion,device)
    

    
    




if __name__ == "__main__":
    main()