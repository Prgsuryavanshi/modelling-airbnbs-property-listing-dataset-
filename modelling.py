import os
import abc
import json
import torch
import joblib
import itertools
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

warnings.filterwarnings('ignore')

from tabular_data import load_airbnb

# split the dataset into train & test data using train_test split()
def get_split_data(features: pd.core.frame.DataFrame, label: pd.core.series.Series,task_folder: str) -> tuple[pd.core.frame.DataFrame, pd.core.series.Series, pd.core.frame.DataFrame, pd.core.series.Series, pd.core.frame.DataFrame, pd.core.series.Series]:
    """This function is used to:
       -> get numeric data from dataset and Price_Night/Category as label based on task folder.
       -> normalize the dataset and split it into the train, test and validation dataset.

    Args:
        features (pd.core.frame.DataFrame): training dataset
        label (pd.core.series.Series): target data
        task_folder (str): regression/classification task

    Returns:
        x_train (pd.core.frame.DataFrame): independent training dataset.
        y_train (pd.core.frame.DataFrame): dependent training dataset.
        x_val (pd.core.frame.DataFrame): independent validation dataset.
        y_val (pd.core.frame.DataFrame): dependent validation dataset.
        x_test (pd.core.frame.DataFrame): independent testing dataset.
        y_test (pd.core.frame.DataFrame): dependent testing dataset.
    """
    
    if task_folder=='models/classification':
        # split the scaled_dataset into train+val and test data
        # due to class imbalance, stratify option is used in train_test_split() to have equal 
        # distribution of all output classes
        x_trainval, x_test, y_trainval, y_test = train_test_split(features, label, stratify=label, test_size=0.10)

        # split the trainval into train and val data
        x_train, x_val, y_train, y_val = train_test_split(x_trainval, y_trainval, stratify=y_trainval, test_size=0.15) 
    else:
        # split the scaled_dataset into train+val and test data
        x_trainval, x_test, y_trainval, y_test = train_test_split(features, label, test_size=0.10)

        # split the trainval into train and val data
        x_train, x_val, y_train, y_val = train_test_split(x_trainval, y_trainval, test_size=0.15)
        
    # normalize x_train, x_test & x_val
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    x_val = scaler.transform(x_val)
    # numpy -> dataframe
    x_train = pd.DataFrame(x_train, columns = features.columns)
    x_val = pd.DataFrame(x_val, columns = features.columns)
    x_test = pd.DataFrame(x_test, columns = features.columns)
    
    return x_train, y_train, x_val, y_val, x_test, y_test

def custom_tune_regression_model_hyperparameters(model_class: abc.ABCMeta, split_data: tuple, hyper_params: dict) -> tuple[str, dict, dict]:
    """This function is used to: 
        -> perform custom grid search by hyperparameters tuning.
        -> calculate R2, MSE, RMSE and MAE of train, test and val data, and 
        -> find best model based on validation rmse score.

    Args:
        model_class (abc.ABCMeta): model class.
        split_data (tuple): training, validation, and test sets.
        hyper_params (dict): dictionary of hyperparameters values to get best model based on 'validation_rmse'.
    
    Returns:
        best_model (str): return the model with best score.
        best_hyperparameter_values (dict): return a dictionary of its best hyperparameter values.
        model_score (dict): return a dictionary of its best performance metrics.
    """
    # unpack split_data
    x_train, y_train, x_val, y_val, x_test, y_test = split_data
    # variable to store best val rmse for comparison
    best_val_rmse = float('inf')
    # empty list for performance metrics
    model_score = {"test_r2":[], "train_r2":[], "val_r2":[],"test_RMSE":[], "train_RMSE":[], "validation_RMSE":[], "test_MSE":[], "train_MSE":[],  "val_MSE":[], "test_MAE":[], "train_MAE":[], "val_MAE":[], "Model_Class":[]}
    # unpack key value pairs from hyper_params dict
    keys, values = zip(*hyper_params.items())
    permutations_hyper_params = [dict(zip(keys, v)) for v in itertools.product(*values)]
    # train model with hyper params
    for params in permutations_hyper_params:
        model = model_class(**params)
        model.fit(x_train, y_train)
        
        # predict target using x_test data
        ypred_test = model.predict(x_test)

        # predict target using x_train data
        ypred_train = model.predict(x_train)
        
        # predict target using x_val data
        ypred_val = model.predict(x_val)
        # get performance metrics based on best val_rmse
        val_rmse = mean_squared_error(y_val,ypred_val,squared=False)
        if val_rmse < best_val_rmse:
            # get best model
            best_model = model
            # if new val_rmse is less than best_val_rmse 
            best_val_rmse = val_rmse
            # get model class name
            model_score["Model_Class"] = str(model.__class__.__name__)
            # get best hyper param
            best_hyperparameter_values = params
            # get model score
            model_score["test_r2"] = r2_score(y_test, ypred_test)
            model_score["train_r2"] = r2_score(y_train, ypred_train)
            model_score["val_r2"] = r2_score(y_val, ypred_val)
            model_score["train_MAE"] = mean_absolute_error(y_train,ypred_train)
            model_score["test_MAE"] = mean_absolute_error(y_test,ypred_test)
            model_score["val_MAE"] = mean_absolute_error(y_val,ypred_val)
            model_score["train_MSE"]= mean_squared_error(y_train,ypred_train)
            model_score["test_MSE"]= mean_squared_error(y_test,ypred_test)
            model_score["val_MSE"]= mean_squared_error(y_val,ypred_val)
            model_score["validation_RMSE"] = val_rmse
            model_score["train_RMSE"] = mean_squared_error(y_train,ypred_train,squared=False)
            model_score["test_RMSE"] = mean_squared_error(y_test,ypred_test,squared=False)
    
    return best_model, best_hyperparameter_values, model_score

# Milestone-4, Task-4 & 6
def tune_regression_model_hyperparameters(model_class: abc.ABCMeta, hyper_params: dict, split_data: tuple, validation: int=5) -> tuple[abc.ABCMeta, dict, dict]:
    """This function is used to:
        -> to perform a grid search using SKLearn's GridSearchCV over a reasonable range of hyperparameter values,
        -> train regression model using best hyper params obtained from GridSearchCV() and
        -> get best model performance metrics
        
    Args:
        model_class (abc.ABCMeta): the model class to be optimized by cross-validated grid-search over a parameter grid.
        hyper_params (dict): a dictionary of hyperparameter names mapping to a list of values to be tried.
        split_data (tuple): training, validation, and test sets.
        validation (int, optional): the cross-validation splitting strategy. Defaults to 5.

    Returns:
        trained_model (abc.ABCMeta): the model class trained and tuned using GridSearchCV().
        best_model_param (dict): a dictionary of best hyperparameters on which the model is trained.
        model_score (dict): a dictionary of performance metrics of the model once it is trained and tuned.
    """
    # unpack split_data
    x_train, y_train, x_val, y_val, x_test, y_test = split_data 
    
    #find best parameters using GridSearchCV()
    grid = GridSearchCV(estimator=model_class, param_grid=hyper_params, cv=validation, scoring='neg_root_mean_squared_error', verbose=1, n_jobs=-1)  
    grid_result = grid.fit(x_train, y_train)
    best_model_param = grid_result.best_params_
    
    # get model class and create an instance of model_class
    reg_model = model_class.__class__
    reg_model = reg_model(**best_model_param)
    # train model
    trained_model = reg_model.fit(x_train, y_train)
    # store score in model_score
    model_score = regression_trained_model_metrics(trained_model, split_data)

    return trained_model, model_score, best_model_param

def regression_trained_model_metrics(trained_model: abc.ABCMeta, split_data: tuple) -> dict:
    """This function caculates the performance metrics of the trained model(trained and tuned using GridSearchCV) 
       and append it to the model_score dictionary.

    Args:
        trained_model (abc.ABCMeta): the model class trained and tuned using GridSearchCV.
        split_data (tuple): training, validation, and test sets.

    Returns:
        model_score (dict): a dictionary of performance metrics.
    """
    # unpack split_data
    x_train, y_train, x_val, y_val, x_test, y_test = split_data
    # empty list to store regression model performance metrics
    model_score = {"test_r2":[],
                   "train_r2":[],
                   "val_r2":[],
                   "test_RMSE":[],
                   "train_RMSE":[],
                   "validation_RMSE":[],
                   "test_MSE":[],
                   "train_MSE":[],
                   "val_MSE":[],
                   "test_MAE":[],
                   "train_MAE":[],
                   "val_MAE":[],
                   "Model_Class":[]}
    
    # predict target using x_test data
    ypred_test = trained_model.predict(x_test)

    # predict target using x_train data
    ypred_train = trained_model.predict(x_train)

    # predict target using x_val data
    ypred_val = trained_model.predict(x_val)
    
    # save model_class name
    model_score["Model_Class"].append(str(trained_model.__class__.__name__))
    
    # append performance metrics to model_score
    model_score["test_r2"].append(r2_score(y_test, ypred_test))
    model_score["train_r2"].append(r2_score(y_train, ypred_train))
    model_score["val_r2"].append(r2_score(y_val, ypred_val))
    model_score["train_MAE"].append(mean_absolute_error(y_train,ypred_train))
    model_score["test_MAE"].append(mean_absolute_error(y_test,ypred_test))
    model_score["val_MAE"].append(mean_absolute_error(y_val,ypred_val))
    model_score["train_MSE"].append(mean_squared_error(y_train,ypred_train))
    model_score["test_MSE"].append(mean_squared_error(y_test,ypred_test))
    model_score["val_MSE"].append(mean_squared_error(y_val,ypred_val))
    model_score["validation_RMSE"].append(mean_squared_error(y_val,ypred_val,squared=False))
    model_score["train_RMSE"].append(mean_squared_error(y_train,ypred_train,squared=False))
    model_score["test_RMSE"].append(mean_squared_error(y_test,ypred_test,squared=False))
    
    return model_score

# Milestone-4, Task-5
def save_model(trained_model: abc.ABCMeta, best_model_param: dict, model_score: dict, folder: str):
    """This function saves:
       -> the model in a file called model.joblib (SKLearn regression & classification model) and model.pt(PyTorch)
       -> its hyperparameters in a file called hyperparameters.json.
       -> its performance metrics in a file called metrics.json once it's trained and tuned.
       The function take in the name of a folder where these files are saved as a keyword argument "folder".

    Args:
        trained_model (abc.ABCMeta): the model class trained and tuned using GridSearchCV.
        best_model_param (dict): a dictinary of best hyperparameters on which the model is trained.
        model_score (dict): a dictinary of performance metrics of the model once it is trained and tuned.
        folder (str): the name of a folder where these files are saved.
    """
    # save pytorch model
    if isinstance(trained_model, torch.nn.Module):
        with open(os.path.join(folder, "model.pt"), "wb") as out_pt:
            torch.save(trained_model.state_dict(), out_pt )
    
    # save regression/classification model
    else:
        with open(os.path.join(folder, "model.joblib"), "wb") as out_joblib:
            joblib.dump(trained_model, out_joblib)
    
    # save model hyperparameters
    with open(os.path.join(folder, "hyperparameters.json"), "w") as param_json:
        json.dump(best_model_param, param_json, indent=4)
    
    # save model metrics
    with open(os.path.join(folder, "metrics.json"), "w") as metrics_json:
        json.dump(model_score, metrics_json, indent=4)

# Milestone-4, Task-6
# Milestone-5, Task-5
def classification_trained_model_metrics(trained_model: abc.ABCMeta, split_data: tuple) -> dict:
    """This function caculates the performance metrics of the trained classifier model
    (trained and tuned using GridSearchCV) and append it to the model_score dictionary.

    Args:
        trained_model (abc.ABCMeta): the model class trained and tuned using GridSearchCV.
        split_data (tuple): training, validation, and test sets.

    Returns:
        model_score (dict): a dictionary of performance metrics.
    """
    # unpack split_data
    x_train, y_train, x_val, y_val, x_test, y_test = split_data
    
    # empty list to store classification model performance metrics
    model_score = {"accuracy":{}, "f1_score":{}, "precision":{}, "recall":{}}

    # predict target using x_test data
    ypred_test = trained_model.predict(x_test)

    # predict target using x_train data
    ypred_train = trained_model.predict(x_train)

    # predict target using x_val data
    ypred_val = trained_model.predict(x_val)
    
    # calculate f1 score 
    test_f1 = f1_score(y_test, ypred_test, average='weighted')
    train_f1 = f1_score(y_train, ypred_train, average='weighted')
    val_f1 = f1_score(y_val, ypred_val, average='weighted')
    
    # calculate precision score 
    test_precision = precision_score(y_test, ypred_test, average='weighted')
    train_precision = precision_score(y_train, ypred_train, average='weighted')
    val_precision = precision_score(y_val, ypred_val, average='weighted')
    
    # calculate recall score 
    test_recall = recall_score(y_test, ypred_test, average='weighted')
    train_recall = recall_score(y_train, ypred_train, average='weighted')
    val_recall = recall_score(y_val, ypred_val, average='weighted')
    
    # calculate accuracy    
    test_accuracy = accuracy_score(y_test, ypred_test, normalize=True)
    train_accuracy = accuracy_score(y_train, ypred_train, normalize=True)
    val_accuracy = accuracy_score(y_val, ypred_val, normalize=True)    
    
    # append scores/metrics to model_score
    model_score['accuracy'] = {'train': train_accuracy, 'test': test_accuracy, 'val': val_accuracy}
    model_score['precision'] = {'train': train_precision, 'test': test_precision, 'val': val_precision}
    model_score['recall'] = {'train': train_recall, 'test': test_recall, 'val': val_recall}
    model_score['f1_score']= {'train': train_f1, 'test': test_f1, 'val': val_f1}
    
    # save model_class name
    model_score["Model_Class"] = str(trained_model.__class__.__name__)

    return model_score

def tune_classification_model_hyperparameters(model_class: abc.ABCMeta, hyper_params: dict, split_data: tuple, validation: int=5) -> tuple[abc.ABCMeta, dict, dict]:
    """This function is used to:
        -> to perform a grid search using SKLearn's GridSearchCV over a reasonable range of hyperparameter values,
        -> train classification model using best hyper params obtained from GridSearchCV() and
        -> get best model performance metrics

    Args:
        model_class (abc.ABCMeta): the model class to be optimized by cross-validated grid-search over a parameter grid.
        hyper_params (dict): a dictionary of hyperparameter names mapping to a list of values to be tried.
        split_data (tuple): training, validation, and test sets.
        validation (int, optional): Defaults to 5.

    Returns:
        trained_model (abc.ABCMeta): the model class trained and tuned using GridSearchCV.
        best_model_param (dict): a dictionary of best hyperparameters on which the model is trained.
        model_score (dict): a dictionary of performance metrics of the model once it is trained and tuned.
    """
    # unpack split_data
    x_train, y_train, x_val, y_val, x_test, y_test = split_data
    
    #find best parameters using GridSearchCV()
    grid = GridSearchCV(estimator=model_class, param_grid=hyper_params, cv=validation, scoring='accuracy', verbose=1, n_jobs=-1)  
    grid_result = grid.fit(x_train, y_train)
    best_model_param = grid_result.best_params_
    
    # get model class and create an instance of model_class
    clf_model = model_class.__class__
    clf_model = clf_model(**best_model_param)
    
    # train model
    trained_model = clf_model.fit(x_train, y_train)
    
    # store score in model_score
    model_score = classification_trained_model_metrics(trained_model, split_data)

    return trained_model, model_score, best_model_param

# Milestone-4, Task-6
def evaluate_all_models(task_folder: str, split_data: tuple) -> None:
    """This function evaluate all the performance of the model(regressor and/or classifier) 
       by using different models provided by sklearn:
       -> SGDRegressor(Linear regressor)
       -> LogisticRegression(Linear classifier)
       -> Decision trees (regressor and classifier)
       -> Gradient boosting (regressor and classifier)
       -> Random forests (regressor and classifier)
       Save the model, hyperparameters, and metrics in a folder named after the model class.

    Args:
        task_folder (str): the name of the parent folder where evaluated regressor and classifier models are saved.
        split_data (tuple): training, validation, and test sets.
    """
    # model_class = SGDRegressor()
    sgd_reg_parameters = {
        'loss'         : ['squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
        'penalty'      : ['l1', 'l2', 'elasticnet', None],
        'alpha'        : [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
        'epsilon'      : [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        'eta0'         : [1, 10, 100],
        'max_iter'     : [0, 0.5, 1, 10, 100, 1000, 5000, 10000]
    }
    # model_class = DecisionTreeRegressor()
    decisiontree_reg_parameters = {
        'criterion'               : ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
        'splitter'                : ['best', 'random'],
        'max_depth'               : [1,3,5,7,9,11,12, None],
        'min_samples_leaf'        : [1,2,3,4,5,6,7,8,9,10],
        'min_weight_fraction_leaf': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
        'max_features'            : ["log2","sqrt",None],
        'max_leaf_nodes'          : [None,10,20,30,40,50,60,70,80,90]
    }
    # model_class = GradientBoostingRegressor()
    gradientboosting_reg_parameters = {
        'learning_rate': [0.01,0.04],
        'n_estimators' : [100, 1500],
        'subsample'    : [0.9, 0.1],
        'max_depth'    : [4,10, None]
    }
    # model_class = RandomForestRegressor()
    randomforest_reg_parameters = {
        'n_estimators'     : [5,20,50,100], # number of trees in the random forest
        'max_features'     : ['sqrt'], # number of features in consideration at every split
        'min_samples_split': [2, 6, 10], # minimum sample number to split a node
        'min_samples_leaf' : [1, 3, 4], # minimum sample number that can be stored in a leaf node
        'bootstrap'        : [True, False] # method used to sample data points
    }
    # dictionary of model class and hyperparameters
    regressor_models = {
            'linear_regression' : [SGDRegressor(), sgd_reg_parameters],
            'decision_tree'     : [DecisionTreeRegressor(), decisiontree_reg_parameters],
            'gradient_boosting' : [GradientBoostingRegressor(), gradientboosting_reg_parameters],
            'random_forests'    : [RandomForestRegressor(), randomforest_reg_parameters],
        }

    # model_class = LogisticRegression()
    sgd_clf_parameters = {
        'solver'         : ['newton-cg', 'sag', 'saga', 'lbfgs'],
        'penalty'        : ['l2','none'],
        'C'              : [1.0],
        'max_iter'       : [100, 1000,2500, 5000]
    }
    # model_class = DecisionTreeClassifier()
    decisiontree_clf_parameters = {
        'criterion'               : ["gini","entropy", "log_loss"],
        'splitter'                : ['best', 'random'],
        'max_depth'               : [1,3,5,7,12, None],
        'min_samples_leaf'        : [1,3,4,5,7,8,10],
        'min_weight_fraction_leaf': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
        'max_features'            : ["log2","sqrt",None],
        'max_leaf_nodes'          : [None,10,20,30,40,50,60,70,80,90],
        'min_impurity_decrease'   : [0.0],
        'ccp_alpha'               : [0.0]
    }
    # model_class = GradientBoostingClassifier()
    gradientboosting_clf_parameters = {
        'loss'              : ['log_loss', 'deviance', 'exponential'],
        'learning_rate'     : [0.01,0.04],
        'n_estimators'      : [100, 1500],
        'subsample'         : [0.9, 0.1],
        'min_samples_split' : [2],
        'max_features'      : ['sqrt', 'log2'],
        'min_samples_leaf'  : [1, 3, 4],
        'max_depth'         : [4,10, None]
    }
    # model_class = RandomForestClassifier()
    randomforest_clf_parameters = {
        'n_estimators'             : [5,20,50,100], # number of trees in the random forest
        'criterion'                : ["gini","entropy", "log_loss"],
        'max_depth'                : [1,3,5,7, None],
        'max_features'             : ['log2', 'sqrt'], # number of features in consideration at every split
        'min_samples_split'        : [2], # minimum sample number to split a node
        'min_samples_leaf'         : [1, 3, 4], # minimum sample number that can be stored in a leaf node
        'bootstrap'                : [True, False], # method used to sample data points
        'min_weight_fraction_leaf' : [0.1,0.4,0.5,0.6,0.9],
        'max_leaf_nodes'           : [None,10,20,60,70,80,90],
        'min_impurity_decrease'    : [0.0],
        'oob_score'                : [False],
        'ccp_alpha'                : [0.0]
    }
    # dictionary of model class and hyperparameters
    classifier_models = {
        'logistic_regression' : [LogisticRegression(), sgd_clf_parameters],
        'decision_tree'       : [DecisionTreeClassifier(), decisiontree_clf_parameters],
        'gradient_boosting'   : [GradientBoostingClassifier(), gradientboosting_clf_parameters],
        'random_forests'      : [RandomForestClassifier(), randomforest_clf_parameters],
    }
    
    # select regression or classification based on task_folder
    if task_folder == 'models/regression':
        for key in regressor_models:
            trained_model, model_score, best_model_param = tune_regression_model_hyperparameters(regressor_models[key][0],regressor_models[key][1], split_data, validation=2)
            save_model(trained_model, best_model_param, model_score, folder=os.path.join(task_folder,key))
    
    elif task_folder == 'models/classification':
        for key in classifier_models:
            trained_model, model_score, best_model_param = tune_classification_model_hyperparameters(classifier_models[key][0],classifier_models[key][1], split_data, validation=2)
            save_model(trained_model, best_model_param, model_score, folder=os.path.join(task_folder,key))
    else:
        pass

    return

# Milestone-4, Task-7
def find_best_model(task_folder: str) -> tuple[abc.ABCMeta, dict, dict]:
    """This function evaluates which model is best based on validation_rmse(for regression)/accuracy(for classification) 
    & returns:
       -> the loaded model.
       -> a dictionary of its hyperparameters.
       -> a dictionary of its performance metrics.

    Args:
        task_folder (str): the name of the parent folder where all models metrics are compared to find the best model.
    
    Returns:
        model (abc.ABCMeta): the model class.
        performance_metrics (dict): return a dictionary of its performance metrics.
        hyperparameters (dict): return a dictionary of its hyperparameter values.
    """
    # an empty list to store the data frames
    list_of_json_df = [] 
    path_of_the_directory = Path(task_folder)
    # get data from json files in the 'path_of_the_directory' location
    # store it in dataframe
    # ignore .DS Store, *.joblib and 'neural_networks' folder
    for filename in os.listdir(path_of_the_directory):
        f = os.path.join(path_of_the_directory,filename)
        if os.path.isfile(f):
            pass
        elif filename == 'neural_networks':
            pass
        else:
            json_path = os.path.join(f,'metrics.json') 
            with open(json_path) as json_file:
                # read data frame from json file
                data = pd.read_json(json_file) 
                print(data)
                # append the path to the data frame
                data['Path'] = f
                # append the data frame to the list
                list_of_json_df.append(data)       
    # concatenate all the data frames in the list.
    df_metrics = pd.concat(list_of_json_df, ignore_index=True)
    print(df_metrics)
    # Select the path of the directory for the model with best metrics
    # Regression
    if task_folder == 'models/regression':
        best_model_path = df_metrics[df_metrics.validation_RMSE==df_metrics.validation_RMSE.min()].Path.values[0]
    # Classification
    elif task_folder == 'models/classification':
        # Get validation accuracy score for each model, which is at every other index+2 starting from 0
        acc_result = df_metrics.loc[2::2, 'accuracy'].values
        path_result = df_metrics.loc[2::2, 'Path'].values
        # Find the index corresponding to max validation score
        max_index = acc_result.argmax()
        # Find the path corresponding to max validation score
        best_model_path = path_result[max_index]
    else:
        pass
    # Load the best model, a dictionary of its hyperparameters
    # and a dictionary of its performance metrics.
    for filename in os.listdir(best_model_path):
        if filename=='metrics.json':
            with open(os.path.join(best_model_path,'metrics.json')) as json_file:
                performance_metrics = json.load(json_file)
        elif filename=='hyperparameters.json':
            with open(os.path.join(best_model_path,'hyperparameters.json')) as json_file1:
                hyperparameters = json.load(json_file1)
        elif filename=='model.joblib':
            model = joblib.load(os.path.join(best_model_path,'model.joblib'))   
        else:
            pass
        
    return model, performance_metrics, hyperparameters



if __name__ == "__main__":

    dataset = pd.read_csv("clean_tabular_data.csv")

    # tune, train, save and find best regression model
    features,label = load_airbnb(dataset, label='Price_Night', numeric=True)
    task_folder='models/regression'
    split_data = get_split_data(features, label,task_folder='models/regression')
    evaluate_all_models(task_folder, split_data)
    best_reg_model = find_best_model(task_folder)
    print(best_reg_model)
    
    # tune, train, save and find best classification model
    features,label = load_airbnb(dataset, label='Category', numeric=True)
    task_folder='models/classification'
    split_data = get_split_data(features, label,task_folder='models/classification')
    evaluate_all_models(task_folder, split_data)
    best_clf_model = find_best_model(task_folder)
    print(best_clf_model)



'''
    #Custom Tune Regression Model Hyperparameters
    #Comment out code above to run custom_tune_regression_model_hyperparameters()

    # Milestone-4, Task-3
    # define hyperparameter values
    model_class = SGDRegressor
    loss = ['squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']
    penalty = ['l1', 'l2', 'elasticnet', None]
    alpha = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    learning_rate = ['constant', 'optimal', 'invscaling', 'adaptive']
    epsilon = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    eta0 = [1, 10, 100]
    max_iter = [0, 0.5, 1, 10, 100, 1000, 5000, 10000]

    hyper_params = dict(loss=loss, penalty=penalty, alpha=alpha, learning_rate=learning_rate, epsilon=epsilon, eta0=eta0)

    dataset = pd.read_csv("clean_tabular_data.csv")
    features,label = load_airbnb(dataset, label='Category', numeric=True)
    #Select task based on classification or regression
    task_folder='models/classification'
    #task_folder='models/regression'
    #split_data = get_split_data(features, label,task_folder)

    best_model, best_hyperparameter_values, model_score = custom_tune_regression_model_hyperparameters(model_class,split_data,hyper_params)

'''