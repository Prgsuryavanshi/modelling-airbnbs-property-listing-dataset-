# Tune the model

# Define a function generate_nn_configs() which will create many config dictionaries for network.

# Define a function called find_best_nn() which will call this function(generate_nn_configs()) and then sequentially trains models with each config.

# This function should save the config used in the hyperparameters.json file for each model trained and returns the best model, metrics and hyperparameters.

# Save the best model in folder.

# Try 16 different parameterisations of the network.

import abc
import torch
from torch import nn
import pandas as pd
import numpy as np
import warnings
import joblib
import os
import yaml
import sys
import time
import datetime
import json
import torch.utils.data.dataloader as D
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import mean_squared_error, r2_score

eps = sys.float_info.epsilon
warnings.filterwarnings('ignore')

dataset = pd.read_csv("clean_tabular_data.csv")

# function to split and normalize the dataset
def get_split_data(dataset: pd.core.frame.DataFrame) -> tuple[pd.core.frame.DataFrame, pd.core.series.Series, pd.core.frame.DataFrame, pd.core.series.Series, pd.core.frame.DataFrame, pd.core.series.Series]:
    """This function is used to:
       -> get numeric data from dataset and Price_Night as label.
       -> normalize the dataset and split it to the train, test and validation dataset.

    Args:
        dataset (pd.core.frame.DataFrame): dataset

    Returns:
        x_train_data (pd.core.frame.DataFrame): independent training dataset.
        y_train_data (pd.core.frame.DataFrame): dependent training dataset.
        x_val_data (pd.core.frame.DataFrame): independent validation dataset.
        y_val_data (pd.core.frame.DataFrame): dependent validation dataset.
        x_test_data (pd.core.frame.DataFrame): independent testing dataset.
        y_test_data (pd.core.frame.DataFrame): dependent testing dataset.
    """
    # extract numerical data from dataset
    dataset = dataset.select_dtypes(include='number')
    
    # save target data in label
    label = dataset['Price_Night']
    
    # drop label column from dataset and saved it to the features
    features = dataset.drop('Price_Night', axis=1)
    
    # normalize the features dataset and saved it to the scaled_dataset
    scaler = MinMaxScaler()
    scaled_dataset = scaler.fit_transform(features)
    scaled_dataset = pd.DataFrame(scaled_dataset, columns=features.columns)
    
    # split the scaled_dataset into (train & val) and test data
    x, x_test_data, y, y_test_data = train_test_split(scaled_dataset, label, test_size=0.10)
    # split the scaled_dataset into train and val data
    x_train_data, x_val_data, y_train_data, y_val_data = train_test_split(x, y, test_size=0.15)
    
    return x_train_data, y_train_data, x_val_data, y_val_data, x_test_data, y_test_data

# class to convert the dataset into pytorch dataset
class AirbnbNightlyPriceImageDataset(Dataset):
    """This class is used to create a pytorch dataset.

    Args:
        Dataset (pd.core.frame.DataFrame): cleaned dataset.
    """
    def __init__(self, dataframe: pd.core.frame.DataFrame, label: pd.core.series.Series):
        """This init method is initialising the attributes:
           label: target variable
           features: numeric dataset

        Args:
            dataframe (pd.core.frame.DataFrame): cleaned dataset.
            label (pd.core.series.Series): target variable.
        """
        self.label = label
        self.features = dataframe
        self.features = self.features.select_dtypes(include='number')

    def __getitem__(self, index: int) -> tuple[torch.Tensor, np.int64]:
        """The __getitem__ function loads and returns a sample from the 
            dataset at the given index idx.

        Args:
            index (int): get data based on the index.

        Returns:
            data (torch.Tensor): converted data into tensor.
            target (np.int64): float(datatype) target data.
        """
        data = self.features.iloc[index].values
        target = self.label.iloc[index]       
        data = torch.tensor(data)
        data = data.to(torch.float32)
        
        return data, target

    def __len__(self) -> int:
        """The __len__ function returns the number of samples in our dataset.

        Returns:
            shape of the features dataset. 
        """
        
        return self.features.shape[0]

# function to create a dataloader using pytorch dataset
def get_dataloader(split_data: tuple) -> tuple[D.DataLoader, D.DataLoader, D.DataLoader]:
    """This function is used to get pytorch dataset and creates a dataloader for the train set,
        test set and validation set.

    Args:
        split_data (tuple): it is a tuple of x_train_data, y_train_data, 
                           x_val_data, y_val_data, x_test_data, y_test_data.

    Returns:
        train_dataloader (D.DataLoader): dataloader for training set.
        test_dataloader (D.DataLoader): dataloader for test set.
        val_dataloader (D.DataLoader): dataloader for validation set.
    """
    # unpack split data
    x_train_data, y_train_data, x_val_data, y_val_data, x_test_data, y_test_data = split_data
    # get train, test and val pytorch dataset
    train_dataset = AirbnbNightlyPriceImageDataset(x_train_data, y_train_data)
    test_dataset = AirbnbNightlyPriceImageDataset(x_test_data, y_test_data)
    val_dataset = AirbnbNightlyPriceImageDataset(x_val_data, y_val_data)
    # get train, test and val pytorch dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True, drop_last=False)
    
    return train_dataloader, test_dataloader, val_dataloader

# function to define layers
class LinearRegressionModel(torch.nn.Module):
    """This class is used to define layers.

    Args:
        torch (class): base class from torch which is inherited.
    """
    def __init__(self,config: dict):
        """This init function is used to initialise the layers

        Args:
            config (dict): dictionary of parameters.
        """
        super().__init__()
        # define hyperparameters
        self.input_node_size = config["input_node_size"]
        self.output_node_size = config["output_node_size"]
        self.hidden_node_size = config["hidden_layer_width"]
        self.model_depth = config["model_depth"]
        # define layers
        self.layers = torch.nn.Sequential()
        
        for i in range(self.model_depth):
            if i==0:
                # input layer
                self.layers.append(torch.nn.Linear(self.input_node_size, self.hidden_node_size))
                self.layers.append(torch.nn.Softplus())
            else:
                # hidden layer
                self.layers.append(torch.nn.Linear(self.hidden_node_size, self.hidden_node_size))
                self.layers.append(torch.nn.Softplus())
        # output layer     
        self.layers.append(torch.nn.Linear(self.hidden_node_size, self.output_node_size))
               
    def forward(self, features: pd.core.frame.DataFrame) -> torch.Tensor:
        """This function is used to computes output tensor from input tensor.
           It takes in a tensor (features) and passes it through the operations 
           as defined in the __init__ method.

        Args:
            features (pd.core.frame.DataFrame): features dataset

        Returns:
            torch.Tensor: prediction
        """
        # return prediction
        return self.layers(features)

# function to load yaml configuration file
def get_nn_config(config_name: str) -> dict:
    """This function is used to read the yaml file.

    Args:
        config_name (str): name of the yaml file.

    Returns:
        config (dict): dictionary of hyparameters for tunning.
    """
    with open(os.path.join( "./",config_name)) as file:
        config = yaml.safe_load(file)
        
    return config

# class to calculate RMSE loss
class RMSELoss(nn.Module):
    """This function calculates RMSE loss.

    Args:
        nn.Module (class): base class.
    """
    def __init__(self):
        """This function is initialising the attribute of the class object.
        """
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self, yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """This function is used to computes output tensor from input tensor.
           
        Args:
            yhat (torch.Tensor): predicted target data.
            y (torch.Tensor): actual target data.

        Returns:
            torch.Tensor: RMSE loss
        """

        return torch.sqrt(self.mse(yhat,y)+self.eps)

# dictionary of the empty list of train and val data
loss_stats = {
    'train': [],
    "val": []
}

# function to train and validate the data and save the model to given directory
def train(train_dataloader: D.DataLoader, val_dataloader: D.DataLoader, config: dict, model: abc.ABCMeta, epochs: int=2) -> tuple[abc.ABCMeta, float, str]:
    """This function is used to: 
       -> train and validate the data using optimiser.
       -> Use tensorboard to visualize the training curves of the model 
          and the accuracy both on the training and validation set.
       -> forward and backward propagation.
       -> save the model in a given directory with current date and time.
       -> calculate loss. 

    Args:
        train_dataloader (D.DataLoader): dataloader for training set.
        val_dataloader (D.DataLoader): dataloader for validation set.
        config (dict): dictionary of parameters.
        model (abc.ABCMeta): pytorch neural network model for training.
        epochs (int): training all the neural network for the given cycle. Defaults to 2.

    Returns:
        model (abc.ABCMeta): trained model.
        training_duration (float): time taken during training the data.
        directory (str): folder where model will be saved.
    """
    if config['optimiser'] == 'sgd':
        optimiser = torch.optim.SGD(model.parameters(), lr=config['learning_rate'])
    elif config['optimiser'] == 'adagrad':
        optimiser = torch.optim.Adagrad(model.parameters(),lr=config['learning_rate'])
    else:
        optimiser = torch.optim.Adam(model.parameters(),lr=config['learning_rate'])
        
    writer = SummaryWriter()   
    batch_idx = 0
    
    # Get the current date and time
    now = datetime.datetime.now()
    current_date_time = now.strftime("%Y-%m-%d_%H:%M:%S")

    # Create a new directory with the current date and time
    directory = os.path.join('models/neural_networks_models/regression/', current_date_time)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    training_duration = 0
    for epoch in range(epochs):
        
        start_time = time.time()
        
        model.train()
        train_epoch_loss = 0
        for batch in train_dataloader:
            features, label = batch
            label = label.to(torch.float32)
            prediction = model(features)
            loss = F.mse_loss(prediction,label)
            loss.backward()
            
            # optimisation step
            optimiser.step()
            # reset grad to zero
            optimiser.zero_grad()
            
            writer.add_scalar('loss', loss.item(), batch_idx)
            batch_idx += 1
            train_epoch_loss += loss.item()
        end_time = time.time()
        training_duration += end_time - start_time
        
        #### Delete this code in the final VScode update
        ### Added for jupyter only
        # Validation Step
        # set the model to eval mode
        model.eval()
        val_epoch_loss = 0
        # turn off gradients for validation
        with torch.no_grad():
            for batch in val_dataloader:
                features, label = batch
                label = label.to(torch.float32)
                prediction = model(features)
                loss = F.mse_loss(prediction,label) 
                # accumulate the valid_loss
                val_epoch_loss += loss.item()
                
        # Print val and train loss
        train_epoch_loss /= len(train_dataloader)
        val_epoch_loss /= len(val_dataloader)
        
        loss_stats['train'].append(train_epoch_loss)
        loss_stats['val'].append(val_epoch_loss)

        print(f'Epoch {epoch+0:03}: | Train MSE: {train_epoch_loss:.2f} | Val MSE: {val_epoch_loss:.2f}')
    
    return model, training_duration, directory

# function to calculate the performance metrics
def model_metrics_nn(model: abc.ABCMeta, x_data: torch.Tensor , y_data: torch.Tensor) -> tuple[np.float64, np.float64]:
    """This function calculate the performance metrics.

    Args:
        model (abc.ABCMeta): trained model
        x_data (torch.Tensor): feature data.
        y_data (torch.Tensor): target data.

    Returns:
        rmse (np.float64): RMSE loss
        r2 (np.float64): r2 score
    """
    criterion = RMSELoss()
    model.eval()
    with torch.no_grad():
        r2 = 0
        x_data = torch.tensor(x_data.values).to(torch.float32)
        y_data = torch.tensor(y_data.values).to(torch.float32)
        prediction = model(x_data)
        rmse = criterion(prediction,y_data)
        rmse = np.float64(rmse)
        
        y_data = y_data.detach().numpy()
        prediction = prediction.detach().numpy()
        r2 += r2_score(y_data,prediction)

    return rmse, r2  

# function to save the model after being trained and tuned
def save_model(trained_model: abc.ABCMeta, best_model_param: dict, model_score: dict, folder: str):
    """This function saves:
       -> the model in a file called model.pt.
       -> its hyperparameters in a file called hyperparameters.json.
       -> its performance metrics in a file called metrics.json once.

    Args:
        trained_model (abc.ABCMeta): trained model.
        best_model_param (dict): best model parameters.
        model_score (dict): best model score.
        folder (str): name of the directory.
    """
    if isinstance(trained_model, torch.nn.Module):
        with open(os.path.join(folder, "model.pt"), "wb") as out_pt:
            torch.save(trained_model.state_dict(), out_pt )
    else:
        with open(os.path.join(folder, "model.joblib"), "wb") as out_joblib:
            joblib.dump(trained_model, out_joblib)
            
    with open(os.path.join(folder, "hyperparameters.json"), "w") as param_json:
        json.dump(best_model_param, param_json, indent=4)
    with open(os.path.join(folder, "metrics.json"), "w") as metrics_json:
        json.dump(model_score, metrics_json, indent=4)

# function to create the get of parameters and save it to the list configs 
def generate_nn_configs() -> list:
    """This function takes the hyperparameters from config file

    Returns:
        configs (list): list of parameters.
    """
    configs = []
    config = get_nn_config("nn_config.yaml")
    # Define different combinations of hyperparameters
    for opt in config['optimiser']:
        for lr in config['learning_rate']:
            for hls in config['hidden_layer_width']:
                for md in config['model_depth']:
                    configs.append({
                        'optimiser': opt,
                        'learning_rate': lr,
                        'hidden_layer_width': hls,
                        'model_depth': md,
                        'input_node_size': config['input_node_size'],
                        'output_node_size': config['output_node_size']
                    })
    
    return configs

# function to find best model after trained, tuned and saved it to the given directory
def find_best_nn(train_dataloader: D.DataLoader, val_dataloader: D.DataLoader, split_data):
    """This function finds the best model and save it to the give directory after being trained and tuned.

    Args:
        train_dataloader (D.DataLoader): training set dataloader.
        val_dataloader (D.DataLoader): validation set dataloader
        split_data (tuple): tuple of dataset after being splited.
    """
    # Generate configs
    configs = generate_nn_configs()
    
    # Initialize variables to keep track of best model
    x_train_data, y_train_data, x_val_data, y_val_data, x_test_data, y_test_data = split_data
    best_rmse = float('inf')
    best_config = None
    best_model = None
    epochs = 20
    model_metrics_dict = {"RMSE_loss": [], "R_squared": [], "training_duration": [],
                          "inference_latency": []}

    for config in configs:
        # Define model
        trained_model, training_duration, directory = train(train_dataloader, val_dataloader, config, model=LinearRegressionModel(config), epochs=epochs)
        
        train_rmse , train_r2 = model_metrics_nn(trained_model, x_train_data, y_train_data)
        test_rmse , test_r2 = model_metrics_nn(trained_model, x_test_data, y_test_data)
        val_rmse , val_r2 = model_metrics_nn(trained_model, x_val_data, y_val_data)
        
        # Append dict of performance metrics
        model_metrics_dict['RMSE_loss'] = {'train': train_rmse, 'test': test_rmse, 'val': val_rmse}
        model_metrics_dict['R_squared'] = {'train': train_r2, 'test': test_r2, 'val': val_r2}
        model_metrics_dict['training_duration'] = training_duration
        model_metrics_dict['inference_latency'] = training_duration/(epochs*len(x_train_data))
        hyperparameters_nn=config
        save_model(trained_model, hyperparameters_nn, model_metrics_dict, folder = directory)
        
        if test_rmse < best_rmse:
            best_rmse = test_rmse
            save_model(trained_model, hyperparameters_nn, model_metrics_dict, folder = 'models/regression/neural_networks')
        else:
            pass

split_data = get_split_data(dataset)

train_dataloader, test_dataloader, val_dataloader = get_dataloader(split_data)

find_best_nn(train_dataloader, val_dataloader,split_data)