import os
import abc
import sys
import yaml
import time
import torch
import json
import joblib
import warnings
import datetime
import pandas as pd
import numpy as np
from torch import nn
import torch.nn.functional as F
from sklearn import preprocessing
import torch.utils.data.dataloader as D
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import mean_squared_error, r2_score, f1_score, precision_score, recall_score, accuracy_score, classification_report

eps = sys.float_info.epsilon
warnings.filterwarnings('ignore')

from tabular_data import load_airbnb

# for the Airbnb dataset, the output labels are from 1 to 10 except 9
# this needs to change because PyTorch supports labels starting from 0. That is [0, n]
# output labels need to be remapped so that it starts from 0
# this can also be implemented using preprocessing.LabelEncoder()
def reverse_mapping(dataset: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    """This function is used to remap the output label to starts from 0.

    Args:
        dataset (pd.core.frame.DataFrame): cleaned dataset.

    Returns:
        dataset (pd.core.frame.DataFrame): remapped dataset.
    """
    class2idx = {
        1:0,
        2:1,
        3:2,
        4:3,
        5:4,
        6:5,
        7:6,
        8:7,
        10:8
    }
    
    # replace bedroom label with new target labels
    dataset['bedrooms'] = dataset['bedrooms'].replace(class2idx)
    
    # remap any output label above 4 (which was label encoded for 5 bedrooms) to 4
    # total count of bedroom no. 6, 7, 8 & 10 is 8 out of 830 data
    dataset['bedrooms'] = dataset['bedrooms'].apply(lambda x: 4 if x >= 5 else x)
    
    return dataset

# data preprocessing of output label for both nn_classification modelling & nn_regression modelling.
def get_processed_data(dataset: pd.core.frame.DataFrame, target: pd.core.series.Series, numeric: bool) -> tuple[pd.core.frame.DataFrame, pd.core.series.Series]:
    """This function is used to preprocess tha data of output label for both nn_classification modelling 
       & nn_regression modelling.

    Args:
        dataset (pd.core.frame.DataFrame): cleaned dataset.
        target (pd.core.series.Series): dependent variable i.e output label.
        numeric (bool): for nn_classification numeric will be false as dataset will consist of categorical value.
                        and for nn_regression numeric will be true as dataset will consist of numerical value.

    Returns:
        features (pd.core.frame.DataFrame): dataframe consists of independent value.
        label (pd.core.series.Series): series consist of dependent value.
    """
    if target == 'bedrooms':
        data = reverse_mapping(dataset)
        features, label = load_airbnb(data, 'bedrooms', numeric)
        cols_to_drop = ['ID', 'Title', 'Description', 'Amenities', 'Location', 'url']
        features.drop(cols_to_drop, axis=1, inplace=True)

        # data in 'Category' has no numerical ordering therefore this column is nominal data
        # perform one hot encoding to the category column and 
        # drop the 'Category' and 'Category_Amazing pools' columns 
        dummies = pd.get_dummies(features.Category, prefix='Category')
        features = pd.concat([features, dummies], axis=1)
        features.drop(['Category', 'Category_Amazing pools'], axis=1, inplace=True)
        
    else:
        features, label = load_airbnb(dataset, 'Price_Night', numeric)
        
    return features,label

# split the data i.e 'features' into train+val and test set.
def get_split_data(features: pd.core.frame.DataFrame, label: pd.core.series.Series, task: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """This function is used to split the data i.e 'features' into train+val and test set based on given 
       task i.e(regression/classification).

    Args:
        features (pd.core.frame.DataFrame): dataframe consists of independent value.
        label (pd.core.series.Series): series consist of dependent value.
        task (str): it is a string(regression/classification) to use while calling the function.

    Returns:
        x_train (np.ndarray): independent training set.
        y_train (np.ndarray): dependent training set.
        x_val (np.ndarray): independent validation set.
        y_val (np.ndarray): dependent validation set.
        x_test (np.ndarray): independent testing set.
        y_test (np.ndarray): dependent testing set.
    """
    if task == 'regression':
        # split the scaled_dataset into train+val and test data
        x_trainval, x_test, y_trainval, y_test = train_test_split(features, label, test_size=0.10)

        # split the trainval into train and val data
        x_train, x_val, y_train, y_val = train_test_split(x_trainval, y_trainval, test_size=0.15) 
        
    else:
        # split the scaled_dataset into train+val and test data
        # due to class imbalance, stratify option is used in train_test_split() to have equal 
        # distribution of all output classes
        x_trainval, x_test, y_trainval, y_test = train_test_split(features, label, stratify=label, test_size=0.10)

        # split the trainval into train and val data
        x_train, x_val, y_train, y_val = train_test_split(x_trainval, y_trainval, stratify=y_trainval, test_size=0.15) 
    
    # normalize the x_train, x_test & x_val
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    x_val = scaler.transform(x_val)
    
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_val, y_val = np.array(x_val), np.array(y_val)
    x_test, y_test = np.array(x_test), np.array(y_test)
 
    return x_train, y_train, x_val, y_val, x_test, y_test

# this class is for nn_regression to convert the dataset to pytorch dataset.
class AirbnbNightlyPriceImageDataset(Dataset):
    """This class represents a custom PyTorch Dataset for the 
       Airbnb Nightly Price prediction task, based on the given dataframe and labels.
      
    Args:
        Dataset (pd.core.frame.DataFrame):  The features dataframe.

    Attributes:
        label : pd.core.series.Series
            Series containing the label (nightly price) information.
        features : pd.core.frame.DataFrame
            Dataframe containing the feature information of the independent values.

    Methods:
    __getitem__(index: int) -> tuple[torch.Tensor, np.int64]
        Returns the data (features) and target (nightly price) for a given index.
    __len__() -> int
        Returns the length of the dataset (number of samples).
    """
    def __init__(self, dataframe: pd.core.frame.DataFrame, label: pd.core.series.Series):
        self.label = label
        self.features = dataframe

    def __getitem__(self, index: int) -> tuple[torch.Tensor, np.int64]:
        """Return the data (image features) and target (nightly price) for a given index.

        Args:
            index (int): Index of the sample to return.

        Returns:
            data, target[torch.Tensor, np.int64]: Tuple containing the data (features) and 
            target (nightly price).
        """
        data = self.features[index]
        data = torch.tensor(data)
        data = data.to(torch.float32)
        target = self.label[index]       
        
        return data, target

    def __len__(self) -> int:
        """Return the length of the dataset (number of samples).

        Returns:
            self.features.shape[0](int): Length of the dataset.
        """
        
        return self.features.shape[0]

# this class is for nn_classification to convert the dataset to pytorch dataset.
class AirbnbClassificationDataset(Dataset):
    """This class represents a custom dataset for PyTorch and is used to create a
       custom dataset for binary or multi-class classification tasks. The dataset is constructed by 
       passing a dataframe containing the feature data and a series containing the labels.

    Args:
        Dataset (pd.core.frame.DataFrame):  The features dataframe.

    Attributes:
        label : pd.core.series.Series
            Series containing the label (bedrooms) information.
        features : pd.core.frame.DataFrame
            Dataframe containing the feature information i.e independent value.

    Methods:
    __getitem__(index: int) -> tuple[torch.Tensor, np.int64]
        Returns the data (features) and target (bedrooms) for a given index.
    __len__() -> int
        Returns the length of the dataset (number of samples).
    """
    def __init__(self, dataframe: pd.core.frame.DataFrame, label: pd.core.series.Series):
        self.label = label
        self.features = dataframe

    def __getitem__(self, index: int) -> tuple[torch.Tensor, np.int64]:
        """Return the data and label at the given index as a tuple of PyTorch tensors. The data is converted to a 
           float32 tensor and the label is converted to a long tensor.
 
        Args:
            index (int): index of value in dataset.

        Returns:
            data, target[torch.Tensor, np.int64]: Returns the length of the dataset.
        """
        data = self.features[index]
        data = torch.tensor(data)
        data = data.to(torch.float32)
        target = self.label[index]
        target = torch.tensor(target)
        
        return data, target

    def __len__(self) -> int:
        """Return the length of the dataset, which is the number of rows in the feature data.

        Returns:
            self.features.shape[0](int): Length of the dataset.
        """
        
        return self.features.shape[0]

def get_class_distribution(obj: np.ndarray) -> dict:
    """This function calculates the class distribution of a given target value 
       array obj and returns a dictionary with the count of each class.

    Args:
        obj (np.ndarray): a numpy ndarray, the target value array.

    Returns:
        count_dict(dict): a dictionary containing the count of each class in the input target value array.
    """
    count_dict = {
        "bed_1": 0,
        "bed_2": 0,
        "bed_3": 0,
        "bed_4": 0,
        "bed_5": 0,
    }
    
    for i in obj:
        if i == 0: 
            count_dict['bed_1'] += 1
        elif i == 1: 
            count_dict['bed_2'] += 1
        elif i == 2: 
            count_dict['bed_3'] += 1
        elif i == 3: 
            count_dict['bed_4'] += 1
        elif i == 4: 
            count_dict['bed_5'] += 1               
        else:
            pass
            
    return count_dict

# TODO check typing and docstring
# due to class imbalance, we use stratified split to create our train, validation, and test sets.
# it still does not ensure that each mini-batch of our model see’s all our classes
# need to over-sample the classes with less number of values. 
# to do that, we use the WeightedRandomSampler.
def weighted_sampling(train_dataset: AirbnbClassificationDataset,y_train: np.ndarray) -> torch.utils.data.sampler.WeightedRandomSampler:
    """This function performs weighted random sampling on a train dataset of train_dataset 
       and its target values of y_train to balance the classes in the dataset. 
       The output is a WeightedRandomSample object which can be used as a data sampler for PyTorch's DataLoader.

    Args:
        train_dataset (__main__.AirbnbClassificationDataset): the train dataset for which the balanced 
                                                              sampler is required.
        y_train (np.ndarray): a numpy ndarray containing target values for the train dataset.

    Returns:
        weighted_sampler (torch.utils.data.sampler.WeightedRandomSample): object with balanced class weights.
    """
    target_list = []
    for _, t in train_dataset:
        target_list.append(t)

    target_list = torch.tensor(target_list)
    class_count = [i for i in get_class_distribution(y_train).values()]
    class_weights = 1./torch.tensor(class_count, dtype=torch.float) 
    class_weights_all = class_weights[target_list]
    weighted_sampler = WeightedRandomSampler(
        weights=class_weights_all,
        num_samples=len(class_weights_all),
        replacement=True)
    
    return weighted_sampler

# create dataloaders for train, test and validation datasets..
def get_dataloader(split_data: tuple, task: str) -> tuple[D.DataLoader, D.DataLoader, D.DataLoader]:
    """This function creates PyTorch dataloaders for train, test and validation 
       datasets based on the task type (regression or classification).
       For the classification task, the dataloader is over-sampled to handle class imbalance.

    Args:
        split_data (tuple): tuple of train, test and validation data.
        task (str): task type, either 'regression' or 'classification'.

    Returns:
        train_dataloader(D.DataLoader): train dataloaders.
        test_dataloader(D.DataLoader): test dataloaders.
        val_dataloader(D.DataLoader):  validation dataloaders.
    """
    
    # unpack split data
    x_train, y_train, x_val, y_val, x_test, y_test = split_data
    
    if task == 'regression':
        
        # get train, test and val pytorch dataset
        train_dataset = AirbnbNightlyPriceImageDataset(x_train, y_train)
        test_dataset = AirbnbNightlyPriceImageDataset(x_test, y_test)
        val_dataset = AirbnbNightlyPriceImageDataset(x_val, y_val)
        
        # create train, test and val pytorch dataloader
        train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False, drop_last=False)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True, drop_last=False)
        
    else:
        
        # create train, test and val pytorch dataset
        train_dataset = AirbnbClassificationDataset(x_train, y_train)
        test_dataset = AirbnbClassificationDataset(x_test, y_test)
        val_dataset = AirbnbClassificationDataset(x_val, y_val)
        
        # need to over-sample the classes with less number of values, use the WeightedRandomSampler
        weighted_sampler = weighted_sampling(train_dataset,y_train)
    
        # create train, test and val pytorch dataloader
        train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False, drop_last=False, sampler=weighted_sampler)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True, drop_last=False)
    
    return train_dataloader, test_dataloader, val_dataloader

# TODO check typing and docstring
class NNRegressionModel(torch.nn.Module):
    """A class for building a neural network for regression tasks.
       init method initializes the attributes.
       Attributes:
           input_node_size (int): the number of input features.
           output_node_size (int): the number of outputs (1 for regression).
           hidden_node_size (int): the width of the hidden layers.
           model_depth (int): the number of hidden layers.
    Args:
        torch.nn.Module (_type_): the network predictions.
    """
    def __init__(self,config: dict):
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
        """Methods:
          forward: computes a forward pass through the network and returns the predictions.

        Args:
            features (pd.core.frame.DataFrame): a pandas dataframe containing the input features for the network.

        Returns:
            self.layers(features)(torch.nn.Sequential): a sequential container for the layers in the network.
        """
        
        # return prediction
        return self.layers(features)

# TODO check typing and docstring
class NNClassificationModel(torch.nn.Module):
    """NNClassificationModel class implements a PyTorch neural network model for multi-class classification.

    Args:
        torch.nn.Module (_type_): _description_

    Class Attributes:
    input_node_size (int): Number of input features
    output_node_size (int): Number of output classes
    hidden_node_size (int): Number of nodes in hidden layers
    model_depth (int): Number of hidden layers in the model
    layers (torch.nn.Sequential): PyTorch sequential container holding the defined layers of the network.

    Methods:
        forward(features: pd.core.frame.DataFrame) -> torch.Tensor:
        Returns the model prediction for the input features tensor.

    Class Initialization:
        init(config: dict):
        Initializes the NNClassificationModel class.
        Accepts a dictionary config to define the hyperparameters and layers of the network.
    """
    def __init__(self,config: dict):
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
                self.layers.append(torch.nn.ReLU())
            else:
                
                # hidden layer
                self.layers.append(torch.nn.Linear(self.hidden_node_size, self.hidden_node_size))
                self.layers.append(torch.nn.ReLU())
                
        # output layer     
        self.layers.append(torch.nn.Linear(self.hidden_node_size, self.output_node_size))
        self.layers.append(torch.nn.Softmax())
               
    def forward(self, features: pd.core.frame.DataFrame) -> torch.Tensor:
        """forward: computes a forward pass through the network and returns the predictions.

        Args:
            features (pd.core.frame.DataFrame): a pandas dataframe containing the input features for the network.

        Returns:
            self.layers(features)(torch.nn.Sequential): Returns the model prediction for the input features tensor.
        """
        
        # return prediction
        return self.layers(features)

# Function to load yaml configuration file
def get_nn_config(config_name: str) -> dict:
    """This function returns the configuration for a neural network model as a dictionary, 
       loaded from a YAML file.

    Args:
        config_name (str): filename of the YAML file containing the network configuration.

    Returns:
        config(dict): Configuration dictionary loaded from the specified YAML file.
    """
    with open(os.path.join( "./",config_name)) as file:
        config = yaml.safe_load(file)
        
    return config

# TODO check typing and docstring
# class to calculate RMSE loss for nn_regression model.
class RMSELoss(nn.Module):
    """Calculates root mean squared error (RMSE) loss between a predicted output yhat and target output y.

    Args:
        nn.Module(_type_): _description_

    Attributes:
        mse (nn.MSELoss): Mean squared error (MSE) loss calculation module.
        eps (float): Small positive value to avoid division by zero in the RMSE calculation.

    Methods:
        forward (yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        Calculates RMSE loss between yhat and y.

    Returns:
        loss(torch.Tensor): RMSE loss between yhat and y.
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self, yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Calculates RMSE loss between yhat and y.

        Args:
            yhat (torch.Tensor): tensor data
            y (torch.Tensor):  tensor data

        Returns:
            tsqrt(self.mse(yhat,y)+self.eps)(orch.Tensor): RMSE loss
        """

        return torch.sqrt(self.mse(yhat,y)+self.eps)

# this function trains the neural network model either regression or classification task.
def train(train_dataloader: D.DataLoader, val_dataloader: D.DataLoader, config: dict, task: str, model: abc.ABCMeta) -> tuple[abc.ABCMeta, float, str]:
    """This function is used to Train a neural network for either regression or classification.

    Args:
        train_dataloader(D.DataLoader): PyTorch data loader object for training data.
        val_dataloader(D.DataLoader): PyTorch data loader object for validation data.
        config(dict): Dictionary containing hyperparameters for training.
        task(str): Type of task. 'regression' or 'classification'.
        model(abc.ABCMeta):  PyTorch model instance.

    Returns:
        model(abc.ABCMeta): Trained model.
        training_duration(float): mean train loss over all epochs.
        directory(str): directory to store the model.
    """ 
    if config['optimiser'] == 'sgd':
        optimiser = torch.optim.SGD(model.parameters(), lr=config['learning_rate'])
    elif config['optimiser'] == 'adagrad':
        optimiser = torch.optim.Adagrad(model.parameters(),lr=config['learning_rate'])
    else:
        optimiser = torch.optim.Adam(model.parameters(),lr=config['learning_rate'])
        
    writer = SummaryWriter()   
    batch_idx = 0
    batch_idx1 = 0
    training_duration = 0
    
    # Get the current date and time
    now = datetime.datetime.now()
    current_date_time = now.strftime("%Y-%m-%d_%H:%M:%S")
    
    if task == 'regression':
        # Create a new directory with the current date and time
        directory = os.path.join('models', 'neural_networks_models', 'regression', current_date_time)
        if not os.path.exists(directory):
            os.makedirs(directory)
    else:
        # Create a new directory with the current date and time
        directory = os.path.join('models', 'neural_networks_models', 'classification', current_date_time)
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    
    for epoch in range(config['epoch']):
        
        start_time = time.time()
        model.train()
        train_epoch_loss = 0
        train_epoch_r2 = 0
        for batch in train_dataloader:
            features, label = batch
            prediction = model(features)
            if task == 'classification':
                loss = F.cross_entropy(prediction,label.long())
            else:
                label = label.to(torch.float32)
                loss = F.mse_loss(prediction,label)
            
            loss.backward()
            
            # optimisation step
            optimiser.step()
            
            # reset grad to zero
            optimiser.zero_grad()           
            writer.add_scalar('train_loss', loss.item(), batch_idx)
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
        
        # turn off gradient for validation
        with torch.no_grad():
            for batch in val_dataloader:
                features, label = batch
                prediction = model(features)
                
                if task == 'classification':
                    loss = F.cross_entropy(prediction,label.long())
                else:
                    loss = F.mse_loss(prediction,label) 
                writer.add_scalar('val_loss', loss.item(), batch_idx1)
                batch_idx1 += 1
                
                # accumulate the valid_loss
                val_epoch_loss += loss.item()
                
        # Print val and train loss
        train_epoch_loss /= len(train_dataloader)
        val_epoch_loss /= len(val_dataloader)

        print(f'Epoch {epoch+0:03}: | Train MSE: {train_epoch_loss:.2f} | Val MSE: {val_epoch_loss:.2f}')

    return model, training_duration, directory

# computes regression evaluation metrics for a PyTorch neural network.
def nn_metrics_regression(model: abc.ABCMeta, x_data: torch.Tensor , y_data: torch.Tensor) -> tuple[np.float64, np.float64]:
    """This function computes the Root Mean Squared Error (RMSE) and the R-squared (R2) values for a 
       PyTorch neural network model on a given set of input and target data.

    Args:
        model(abc.ABCMeta): The trained PyTorch neural network model.
        x_data(torch.Tensor): The input data tensor.
        y_data(torch.Tensor): The target data tensor.

    Returns:
        rmse(np.float64): rmse score of the trained model.
        r2(np.float64): r2 score of the trained model.
    """  
    criterion = RMSELoss()
    model.eval()
    
    with torch.no_grad():
        r2 = 0
        x_data = torch.tensor(x_data).to(torch.float32)
        y_data = torch.tensor(y_data).to(torch.float32)
        prediction = model(x_data)
        rmse = criterion(prediction,y_data)
        rmse = np.float64(rmse)

        prediction = prediction.detach().numpy()
        y_data = y_data.detach().numpy()
        r2 += r2_score(y_data,prediction)

    return rmse, r2

# neural network classification metrics.
def nn_metrics_classification(model: abc.ABCMeta, x_data: torch.Tensor , y_data: torch.Tensor) -> tuple[np.float64, np.float64, np.float64, np.float64, dict]:
    """This function is used to Calculates classification metrics 
       (accuracy, precision, recall, f1-score) for a PyTorch neural network model.

    Args:
        model(abc.ABCMeta): PyTorch neural network model (instance of ABCMeta).
        x_data(torch.Tensor): input data tensor (torch.Tensor).
        y_data(torch.Tensor): ground truth label tensor (torch.Tensor).

    Returns:
        score_accuracy(np.float64): accuracy score.
        score_precision(np.float64): precision score.
        score_recall(np.float64): recall score.
        score_f1(np.float64): f1-score.
        report(dict): dictionary containing the detailed report 
                      generated by scikit-learn's classification_report function (dict).
    """ 
    model.eval()
    with torch.no_grad():
        x_data = torch.tensor(x_data).to(torch.float32)
        y_data = torch.tensor(y_data).to(torch.float32)
        output = model(x_data)
        prediction = torch.argmax(output, 1)
        
        y_data = y_data.detach().numpy()
        prediction = prediction.detach().numpy()
        score_f1 = f1_score(y_data, prediction, average='weighted')
        score_precision = precision_score(y_data, prediction, average='weighted')
        score_recall = recall_score(y_data, prediction, average='weighted')
        score_accuracy = accuracy_score(y_data, prediction)
        target_names = ['Beds - 1', 'Beds - 2', 'Beds - 3','Beds - 4','Beds - 5 & above']
        report = classification_report(y_data, prediction, target_names=target_names, output_dict=True)

    return score_accuracy, score_precision, score_recall, score_f1, report

def save_model(trained_model: abc.ABCMeta, best_model_param: dict, model_score: dict, folder: str):
    """This function saves the trained model, hyperparameters and the metrics of the model.

    Args:
        trained_model (abc.ABCMeta): the model to be saved (torch.nn.Module or joblib model)
        best_model_param (dict): dictionary of the hyperparameters of the model.
        model_score (dict): dictionary of metrics obtained from the model
        folder (str): the folder to save the model
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

def generate_nn_configs(task: str) -> list:
    """This function generates a list of dictionaries of hyperparameters for a neural network model, 
       based on a task string input of either "regression" or "classification".

    Args:
        task (str): a string representing the task, either "regression" or "classification".

    Returns:
        configs(list): a list of dictionaries, where each dictionary represents 
                       a set of hyperparameters for the neural network model.
    """
    configs = []
    if task == 'regression':
        config = get_nn_config("nn_regression_config.yaml")
    else:
        config = get_nn_config("nn_classification_config.yaml")
        
    # define different combinations of hyperparameters
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
                        'output_node_size': config['output_node_size'],
                        'epoch': config['epoch']
                    })
    
    return configs

def find_best_nn(train_dataloader: D.DataLoader, val_dataloader: D.DataLoader, split_data: tuple, task: str):
    """This function Finds the best neural network model based on the supervised learning task(regression or classification) 
       and returns the best model with performance metrics.

    Args:
        train_dataloader (D.DataLoader): PyTorch DataLoader object for training data
        val_dataloader (D.DataLoader): PyTorch DataLoader object for validation data
        split_data (tuple): Tuple containing the data (x,y) split into train, validation, and test datasets
        task (str): the supervised learning task (either "regression" or "classification").
    """
    
    # generate configs based on supervised learning task(i.e regression/classification)
    configs = generate_nn_configs(task)
    
    # initialize variables to keep track of best model
    x_train, y_train, x_val, y_val, x_test, y_test = split_data
    best_rmse = float('inf')
    best_accuracy = -float('inf')
    
    regression_metrics_dict = {"RMSE_loss": [],
                               "R_squared": [],
                               "training_duration": [],
                               "inference_latency": []}
    
    classification_metrics_dict = {"accuracy": [],
                                   "precision": [],
                                   "recall": [],
                                   "f1_score": [],
                                   "training_duration": [],
                                   "inference_latency": [],
                                   "report": []}
    for config in configs:
        
        # train model
        # Regression NN 
        if task == 'regression':
            trained_model, training_duration, directory = train(train_dataloader, val_dataloader, config, task, model=NNRegressionModel(config))
            
            # train data metrics
            train_rmse , train_r2 = nn_metrics_regression(trained_model, x_train, y_train)
            
            # test data metrics
            start_time = time.time()
            test_rmse , test_r2 = nn_metrics_regression(trained_model, x_test, y_test)
            end_time = time.time()
            test_duration = end_time - start_time
            
            # val data metrics
            val_rmse , val_r2 = nn_metrics_regression(trained_model, x_val, y_val)

            # Append dict of performance metrics
            regression_metrics_dict['RMSE_loss'] = {'train': train_rmse, 'test': test_rmse, 'val': val_rmse}
            regression_metrics_dict['R_squared'] = {'train': train_r2, 'test': test_r2, 'val': val_r2}
            regression_metrics_dict['training_duration'] = training_duration
            regression_metrics_dict['inference_latency'] = test_duration/len(x_test)
            hyperparameters_nn=config
            save_model(trained_model, hyperparameters_nn, regression_metrics_dict, folder = directory)

            if test_rmse < best_rmse:
                best_rmse = test_rmse
                save_model(trained_model, hyperparameters_nn, regression_metrics_dict, folder = os.path.join('models', 'regression', 'neural_networks'))
            else:
                pass
            
        # Classification NN 
        else:
            # train model      
            trained_model, training_duration, directory = train(train_dataloader, val_dataloader, config, task, model=NNClassificationModel(config))

            
            # train data metrics
            train_score= nn_metrics_classification(trained_model, x_train, y_train)

            # test data metrics
            start_time = time.time()
            test_score = nn_metrics_classification(trained_model, x_test, y_test)
            end_time = time.time()

            test_duration = end_time - start_time

            # val data metrics
            val_score = nn_metrics_classification(trained_model, x_val, y_val)


            # Append dict of performance metrics
            classification_metrics_dict['accuracy'] = {'train': train_score[0], 'test': test_score[0], 'val': val_score[0]}
            classification_metrics_dict['precision'] = {'train': train_score[1], 'test': test_score[1], 'val': val_score[1]}
            classification_metrics_dict['recall'] = {'train': train_score[2], 'test': test_score[2], 'val': val_score[2]}
            classification_metrics_dict['f1_score']= {'train': train_score[3], 'test': test_score[3], 'val': val_score[3]}
            classification_metrics_dict['training_duration'] = training_duration
            classification_metrics_dict['inference_latency'] = test_duration/len(x_test)
            classification_metrics_dict['report'] = {'train': train_score[4], 'test': test_score[4], 'val': val_score[4]}
            hyperparameters_nn=config
            save_model(trained_model, hyperparameters_nn, classification_metrics_dict, folder = directory)

            if test_score[0] > best_accuracy:
                best_accuracy = test_score[0]
                save_model(trained_model, hyperparameters_nn, classification_metrics_dict, folder = os.path.join('models', 'classification', 'neural_networks'))
            else:
                pass

if __name__ == "__main__":
    
    dataset = pd.read_csv("clean_tabular_data.csv")

    # For nn Regression model
    target = 'Price_Night'
    numeric = True

    # set the supervised learning task to regression
    task = 'regression'

    # data preprocessing and stored in features(independent data) and label(target data)
    features,label = get_processed_data(dataset, target, numeric=True)

    # here split_data is a tuple where (x_train, y_train, x_val, y_val, x_test, y_test) 
    # is stored by calling get_split_data()
    split_data = get_split_data(features, label, task)

    # here  by calling get_dataloader(), unpack the tuple split_data within function to use the split data further
    train_dataloader, test_dataloader, val_dataloader = get_dataloader(split_data, task)

    # get the best model, it's metrics and the best parameters
    find_best_nn(train_dataloader, val_dataloader, split_data, task)

    # for nn_classification model
    # read the dataset
    dataset = pd.read_csv("clean_tabular_data.csv")

    # for nn classification model 
    target = 'bedrooms'
    numeric = False

    # set the supervised learning task to classification
    task = 'classification'

    # data preprocessing and stored in features(independent data) and label(target data)
    features,label = get_processed_data(dataset, target, numeric=False)

    # here split_data is a tuple where (x_train, y_train, x_val, y_val, x_test, y_test) 
    # is stored by calling get_split_data()
    split_data = get_split_data(features, label, task)

    # here  by calling get_dataloader(), unpack the tuple split_data within function to use the split data further
    train_dataloader, test_dataloader, val_dataloader = get_dataloader(split_data, task)

    # get the best model, it's metrics and the best parameters
    find_best_nn(train_dataloader, val_dataloader, split_data, task)

