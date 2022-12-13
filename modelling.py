import os
import abc
import json
import joblib
import pandas as pd
import itertools
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from tabular_data import load_airbnb
global grid_result
global best_model_score
global best_model_param

dataset = pd.read_csv("clean_tabular_data.csv")
features,labels = load_airbnb(dataset, label='Price_Night')

# assign the features to x and labels to y
x = features
y = labels

# split the dataset into train & test data using train_test split()
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

# Milestone 4 - Task 1
# create an SGDClassifier instance which will have methods to do linear regression 
# fitting by gradient descent
sgdr = SGDRegressor(loss="huber", epsilon=0.2)

# train the model by fit method
sgdr.fit(xtrain, ytrain)

# predict target using xtest data
ypred_test = sgdr.predict(xtest)

# Milestone 4 - Task 2

# calculate R2 Score for ypred_test
print(r2_score(ytest, ypred_test))

# predict target using xtrain data
ypred_train = sgdr.predict(xtrain)

# calculate R2 Score for ypred_train
r2_score(ytrain, ypred_train)

# calculate mean squared error for test data
mean_squared_error(ytest, ypred_test, squared=False)

# calculate mean squared error for train data
mean_squared_error(ytrain, ypred_train, squared=False)

# TODO add typing for xtrain, xtest, ytrain, ytest
loss = ['squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']
penalty = ['l1', 'l2', 'elasticnet', None]
alpha = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
learning_rate = ['constant', 'optimal', 'invscaling', 'adaptive']
epsilon = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
eta0 = [1, 10, 100]
max_iter = [0, 0.5, 1, 10, 100, 1000, 5000, 10000]

hyper_params = dict(loss=loss, penalty=penalty, alpha=alpha, learning_rate=learning_rate, epsilon=epsilon, eta0=eta0)


# Milestone 4 - Task 3
# TODO return best model and use validation
# TODO validation is applicable on training and test data?
# TODO metrics required for both training and test data?
def custom_tune_regression_model_hyperparameters(model_class: abc.ABCMeta, xtrain: pd.core.frame.DataFrame, \
    xtest: pd.core.frame.DataFrame, ytrain: pd.core.series.Series, ytest: pd.core.series.Series, \
        hyper_params: dict, validation: int=5):
    """This function is used to: 
        -> perform custom grid search by hyperparameters tuning.
        -> calculate MSE, RMSE and MAE of train and test data and append it to the model_score(performance metrics)

    Args:
        model_class (abc.ABCMeta): regressor 
        xtrain (pd.core.frame.DataFrame): training set
        xtest (pd.core.frame.DataFrame): test set
        ytrain (pd.core.series.Series): training target set
        ytest (pd.core.series.Series): test target set
        hyper_params (dict): dictionary of hyperparameters values to get best optimal value
        validation (int): cross-validation splitting strategy
    Returns:
        model_score (dict): dictionary of optimal scores
    """ 
    model_score = {"R2_test":[], "R2_train":[], "Parameters":[], "validation_RMSE":[], "train_RMSE":[], "train_MSE":[], "test_MSE":[], "train_MAE":[], "test_MAE":[], "Model_Class":[]};
    keys, values = zip(*hyper_params.items())
    permutations_hyper_params = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    for params in permutations_hyper_params:
        clf = model_class(**params)
        clf.fit(xtrain, ytrain)
        
        # predict target using xtest data
        ypred_test = clf.predict(xtest)

        # predict target using xtrain data
        ypred_train = clf.predict(xtrain)
        
        mae_train = mean_absolute_error(ytrain,ypred_train)
        mae_test = mean_absolute_error(ytest,ypred_test)
        
        mse_train = mean_squared_error(ytrain,ypred_train)
        mse_test = mean_squared_error(ytest,ypred_test)
        
        rmse_train = mean_squared_error(ytrain,ypred_train,squared=False)
        rmse_test = mean_squared_error(ytest,ypred_test,squared=False)
        
        model_score["Model_Class"].append(model_class)
        
        model_score["R2_test"].append(r2_score(ytest, ypred_test))
        model_score["R2_train"].append(r2_score(ytrain, ypred_train))
        
        model_score["Parameters"].append(params)
        
        model_score["train_MAE"].append(mae_train)
        model_score["test_MAE"].append(mae_test)
        
        model_score["train_MSE"].append(mse_train)
        model_score["test_MSE"].append(mse_test)
        
        model_score["validation_RMSE"].append(rmse_test)
        model_score["train_RMSE"].append(rmse_train)
    
    return model_score

# TODO add typing, return and docstring
# TODO return model_class?
def tune_regression_model_hyperparameters(model_class, hyper_params, validation):

    scorers = ['r2','neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_root_mean_squared_error']
    grid = GridSearchCV(estimator=model_class, param_grid=hyper_params, cv=validation, scoring=scorers, verbose=1, n_jobs=-1)  
    grid_result = grid.fit(xtrain, ytrain)

    best_model_score = grid_result.best_score_
    best_model_param = grid_result.best_params_

    return grid_result, best_model_score, best_model_param

# TODO call the tune_regression_model_hyperparameters() and save the func return value in a var?
def save_model(folder: str):
    with open(os.path.join(folder, "model.joblib"), "wb") as out_joblib:
        joblib.dump(grid_result, out_joblib)
    with open(os.path.join(folder, "hyperparameters.json"), "w") as param_json:
        json.dump(best_model_param, param_json, indent=4)
    with open(os.path.join(folder, "metrics.json"), "w") as metrics_json:
        json.dump(best_model_score, metrics_json, indent=4)
