"""
AI-accelerated surrogate model for Aqueous Solubility prediction
Built in an hour or two to aid understanding for an interview - so do not judge the code so harshly!

Data from:
https://www.kaggle.com/datasets/sorkun/aqsoldb-a-curated-aqueous-solubility-dataset

Network architecture inspired by the following papers:
---

Model is simple, a Multi-Layer feedforward

"""

# TODO: Implement graphing of Test / Train loss every epoch to determine optimum epoch training time
# TODO: Experiment with other architectures


import logging
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import time

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
        self.fc2 = nn.Linear(hidden_layers, hidden_layers)
        self.fc3 = nn.Linear(hidden_layers, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def normalise(df):
    """
    Normalises a pandas df.
    Outliers minimised as using min-max normalisation
    It is columnwise
    """
    return (df-df.min())/(df.max()-df.min())


def load_data(csv_path, train_factor=0.8):
    logging.info("loading data")
    df = pd.read_csv(csv_path,header=0)
   
 
    # Shuffle the data
    df = df.sample(frac=1).reset_index(drop=True) # drop=True stops new column of old index

    # Removing the 'G' from atomic group so numeric. Could consider converting to valence shell electrons?
    df["Group"] =  df["Group"].str[1:]

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
    train_input = normalise(train_input)

    test_input = test_df.iloc[:, 7:].astype(float)
    test_input = normalise(test_input)

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

    # matching first dim (rows)
    train_dataset = TensorDataset(train_input,train_aqsol)
    trainloader = DataLoader(train_dataset, batch_size = 2**10)

    test_dataset = TensorDataset(test_input,test_aqsol)
    testloader = DataLoader(test_dataset, batch_size = 1)

    logging.info("data loaded")

    input_layers = train_input.shape[1]

    return trainloader, testloader, input_layers

def train(model, loader, epochs,criterion,optimiser,device):
    loss_list = []
    for epoch in range(epochs):
        model.train()
        logging.info(f"Beginning epoch {epoch+1} of {epochs}")
        current_epoch_losses = []
        for i, (inputs, targets) in enumerate(loader):
            optimiser.zero_grad()
            
            # Forward
            output = model(inputs.to(device))
            loss = criterion(output.squeeze(), targets.to(device).squeeze())
            current_epoch_losses.append(loss)

            # Backwards
            loss.backward()
            optimiser.step()
        
        mean_epoch_loss = sum(current_epoch_losses)/len(current_epoch_losses)
        print(f'mean_loss for epoch {epoch+1}: {mean_epoch_loss}')
        loss_list.append(mean_epoch_loss)
        # can implement save model here w torch.save(model,path) but it trains to plateu faster than model loading would take
    logging.info("Training Done")


def eval(model, loader,criterion,device):
    model.eval()

    for i, (inputs, targets) in enumerate(loader):
        aqsol_pred = model(inputs.to(device))
        after_train = criterion(aqsol_pred.squeeze(),targets.to(device).squeeze())
    print(f'test loss {after_train}')
    

def main():
    # Hyperparams
    train_factor = 0.8
    hidden_layers = 64
    epochs = 1000
    lr = 0.001


    logging.basicConfig(format='%(levelname)s at %(asctime)s\n%(message)s\n', datefmt='%M:%S', level=logging.DEBUG)

    # TODO: Add list of variables that feeds into load_data() so you can choose inputs to drop
    trainloader, testloader, input_layers = load_data("curated-solubility-dataset.csv",train_factor=train_factor)

    model = MLP(input_layers,hidden_layers)

    # Hybrid laptop problems ugh
    for i in range(torch.cuda.device_count()):
        logging.debug(f'device {i}: {torch.cuda.get_device_name(i)}')
        # device 0: Quadro RTX 3000 with Max-Q Design

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.debug(f'Using device {device}')
    model = model.to(device)
    logging.debug("Model sent to device")


    criterion = nn.MSELoss()
    optimiser = optim.SGD(model.parameters(),lr=lr)


    train(model, trainloader, epochs, criterion, optimiser,device)
    eval(model, testloader, criterion,device)
    

if __name__ == "__main__":
    main()