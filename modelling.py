import os
import abc
import json
import joblib
import pandas as pd
import itertools
import warnings
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from pathlib import Path

from tabular_data import load_airbnb

warnings.filterwarnings('ignore')
global grid_result
global best_model_score
global best_model_param

dataset = pd.read_csv("clean_tabular_data.csv")
features,labels = load_airbnb(dataset, label='Price_Night', numeric=True)

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

# Milestone-4, Task-3
# define hyperparameter values
loss = ['squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']
penalty = ['l1', 'l2', 'elasticnet', None]
alpha = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
learning_rate = ['constant', 'optimal', 'invscaling', 'adaptive']
epsilon = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
eta0 = [1, 10, 100]
max_iter = [0, 0.5, 1, 10, 100, 1000, 5000, 10000]

hyper_params = dict(loss=loss, penalty=penalty, alpha=alpha, learning_rate=learning_rate, epsilon=epsilon, eta0=eta0)

def custom_tune_regression_model_hyperparameters(model_class: abc.ABCMeta, xtrain: pd.core.frame.DataFrame, \
    xtest: pd.core.frame.DataFrame, ytrain: pd.core.series.Series, ytest: pd.core.series.Series, \
        hyper_params: dict) -> tuple[str, dict, dict]:
    """This function is used to: 
        -> perform custom grid search by hyperparameters tuning.
        -> calculate MSE, RMSE and MAE of train and test data and append it to the model_score(performance metrics).

    Args:
        model_class (abc.ABCMeta): regressor.
        xtrain (pd.core.frame.DataFrame): training set.
        xtest (pd.core.frame.DataFrame): test set.
        ytrain (pd.core.series.Series): training target set.
        ytest (pd.core.series.Series): test target set.
        hyper_params (dict): dictionary of hyperparameters values to get best optimal value.
    Returns:
        best_model (str): return the model with best score.
        best_hyperparameter_values (dict): return a dictionary of its best hyperparameter values.
        performance_metrics (dict): return a dictionary of its best performance metrics.
    """ 
    model_score = {"R2_test":[], "R2_train":[], "Parameters":[], "validation_RMSE":[], "train_RMSE":[], "train_MSE":[], "test_MSE":[], "train_MAE":[], "test_MAE":[], "Model_Class":[]};
    keys, values = zip(*hyper_params.items())
    permutations_hyper_params = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    for params in permutations_hyper_params:
        model = model_class(**params)
        model.fit(xtrain, ytrain)
        
        # predict target using xtest data
        ypred_test = model.predict(xtest)

        # predict target using xtrain data
        ypred_train = model.predict(xtrain)
        
        mae_train = mean_absolute_error(ytrain,ypred_train)
        mae_test = mean_absolute_error(ytest,ypred_test)
        
        mse_train = mean_squared_error(ytrain,ypred_train)
        mse_test = mean_squared_error(ytest,ypred_test)
        
        rmse_train = mean_squared_error(ytrain,ypred_train,squared=False)
        rmse_test = mean_squared_error(ytest,ypred_test,squared=False)
        
        model_score["Model_Class"].append(str(model.__class__.__name__))
        
        model_score["R2_test"].append(r2_score(ytest, ypred_test))
        model_score["R2_train"].append(r2_score(ytrain, ypred_train))
        
        model_score["Parameters"].append(params)
        
        model_score["train_MAE"].append(mae_train)
        model_score["test_MAE"].append(mae_test)
        
        model_score["train_MSE"].append(mse_train)
        model_score["test_MSE"].append(mse_test)
        
        model_score["validation_RMSE"].append(rmse_test)
        model_score["train_RMSE"].append(rmse_train)
    
    model_score_dataframe = pd.DataFrame(model_score)
    best_model_param = model_score_dataframe[model_score_dataframe.validation_RMSE == model_score_dataframe.validation_RMSE.min()]
    best_model = best_model_param.Model_Class.values[0]
    best_hyperparameter_values = best_model_param.Parameters.values[0]
    performance_metrics = best_model_param.drop(['Parameters', 'Model_Class'], axis=1).to_dict('list')
    
    return best_model, best_hyperparameter_values, performance_metrics

# Milestone-4, Task-4
def tune_regression_model_hyperparameters(model_class: abc.ABCMeta, hyper_params: dict, validation: int=5) -> dict:
    """This function uses SKLearn's GridSearchCV to perform 
        a grid search over a reasonable range of hyperparameter values.

    Args:
        model_class (abc.ABCMeta): the model class to be optimized by cross-validated grid-search over a parameter grid.
        hyper_params (dict): a dictionary of hyperparameter names mapping to a list of values to be tried.
        validation (int, optional): the cross-validation splitting strategy. Defaults to 5.

    Returns:
        best_model_param (dict): parameter setting that gave the best results(based on r2 score) on the hold out data.
    """
    grid = GridSearchCV(estimator=model_class, param_grid=hyper_params, cv=validation, scoring='r2', verbose=1, n_jobs=-1)  
    grid_result = grid.fit(xtrain, ytrain)
    best_model_param = grid_result.best_params_

    return best_model_param

def metrics_trained_model(trained_model: abc.ABCMeta) -> dict:
    """This function caculates the performance metrics of the trained model(trained and tuned using GridSearchCV) 
       and append it to the model_score dictionary.

    Args:
        trained_model (abc.ABCMeta): the model class trained and tuned using GridSearchCV.

    Returns:
        model_score (dict): a dictionary of performance metrics.
    """
    model_score = {"R2_test":[], "R2_train":[], "validation_RMSE":[], "train_RMSE":[], "train_MSE":[], "test_MSE":[], "train_MAE":[], "test_MAE":[]};

    ypred_test = trained_model.predict(xtest)

    # predict target using xtrain data
    ypred_train = trained_model.predict(xtrain)
    
    mae_train = mean_absolute_error(ytrain,ypred_train)
    mae_test = mean_absolute_error(ytest,ypred_test)

    mse_train = mean_squared_error(ytrain,ypred_train)
    mse_test = mean_squared_error(ytest,ypred_test)

    rmse_train = mean_squared_error(ytrain,ypred_train,squared=False)
    rmse_test = mean_squared_error(ytest,ypred_test,squared=False)

    model_score["R2_test"].append(r2_score(ytest, ypred_test))
    model_score["R2_train"].append(r2_score(ytrain, ypred_train))

    model_score["train_MAE"].append(mae_train)
    model_score["test_MAE"].append(mae_test)

    model_score["train_MSE"].append(mse_train)
    model_score["test_MSE"].append(mse_test)

    model_score["validation_RMSE"].append(rmse_test)
    model_score["train_RMSE"].append(rmse_train)
    
    return model_score

# Milestone-4, Task-5
def save_model(trained_model: abc.ABCMeta, best_model_param: dict, model_score: dict, folder: str):
    """This function saves:
       -> the model in a file called model.joblib.
       -> its hyperparameters in a file called hyperparameters.json.
       -> its performance metrics in a file called metrics.json once it's trained and tuned.
       The function take in the name of a folder where these files are saved as a keyword argument "folder".

    Args:
        trained_model (abc.ABCMeta): the model class trained and tuned using GridSearchCV.
        best_model_param (dict): a dictinary of best hyperparameters on which the model is trained.
        model_score (dict): a dictinary of performance metrics of the model once it is trained and tuned.
        folder (str): the name of a folder where these files are saved.
    """
    with open(os.path.join(folder, "model.joblib"), "wb") as out_joblib:
        joblib.dump(trained_model, out_joblib)
    with open(os.path.join(folder, "hyperparameters.json"), "w") as param_json:
        json.dump(best_model_param, param_json, indent=4)
    with open(os.path.join(folder, "metrics.json"), "w") as metrics_json:
        json.dump(model_score, metrics_json, indent=4)

# Milestone-4, Task-6
def sgdr_hyperparameters() -> tuple[dict, abc.ABCMeta]:
    """This function defines the hyperparameters for SGDRegressor.

    Returns:
        hyper_params (dict): dictionary of hyperparameters.
        model_class (abc.ABCMeta): the model class.
    """
    model_class = SGDRegressor()
    loss = ['squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']
    penalty = ['l1', 'l2', 'elasticnet', None]
    alpha = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    learning_rate = ['constant', 'optimal', 'invscaling', 'adaptive']
    epsilon = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    eta0 = [1, 10, 100]
    max_iter = [0, 0.5, 1, 10, 100, 1000, 5000, 10000]
    hyper_params = dict(loss=loss, penalty=penalty, alpha=alpha, learning_rate=learning_rate, epsilon=epsilon, eta0=eta0, max_iter=max_iter)
    
    return hyper_params, model_class

def dtr_hyperparameters() -> tuple[dict, abc.ABCMeta]:
    """This function defines the hyperparameters for DecisionTreeRegressor.

    Returns:
        hyper_params (dict): dictionary of hyperparameters.
        model_class (abc.ABCMeta): the model class.
    """
    model_class = DecisionTreeRegressor()
    criterion = ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
    splitter = ['best', 'random']
    max_depth = [1,3,5,7,9,11,12, None]
    min_samples_leaf = [1,2,3,4,5,6,7,8,9,10]
    min_weight_fraction_leaf = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    max_features = ["auto","log2","sqrt",None]
    max_leaf_nodes = [None,10,20,30,40,50,60,70,80,90]
    hyper_params = dict(criterion=criterion, splitter=splitter, max_depth=max_depth, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features, max_leaf_nodes=max_leaf_nodes)
    
    return hyper_params, model_class

def gbr_hyperparameters() -> tuple[dict, abc.ABCMeta]:
    """This function defines the hyperparameters for GradientBoostingRegressor.

    Returns:
        hyper_params (dict): dictionary of hyperparameters.
        model_class (abc.ABCMeta): the model class.
    """
    model_class = GradientBoostingRegressor()
    learning_rate = [0.01,0.04]
    n_estimators = [100, 1500]
    subsample = [0.9, 0.1]
    max_depth = [4,10, None]
    hyper_params = dict(learning_rate=learning_rate, n_estimators=n_estimators, subsample=subsample, max_depth=max_depth)
    
    return hyper_params, model_class

def rfr_hyperparameters() -> tuple[dict, abc.ABCMeta]:
    """This function defines the hyperparameters for RandomForestRegressor.

    Returns:
        hyper_params (dict): dictionary of hyperparameters.
        model_class (abc.ABCMeta): the model class.
    """
    model_class = RandomForestRegressor()
    n_estimators = [5,20,50,100] # number of trees in the random forest
    max_features = ['auto', 'sqrt'] # number of features in consideration at every split
    min_samples_split = [2, 6, 10] # minimum sample number to split a node
    min_samples_leaf = [1, 3, 4] # minimum sample number that can be stored in a leaf node
    bootstrap = [True, False] # method used to sample data points
    hyper_params = dict(n_estimators=n_estimators, max_features=max_features, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, bootstrap=bootstrap)
    
    return hyper_params, model_class

def evaluate_all_models() -> None:
    """This function evaluate all the performance of the model by using different models provided by sklearn.
       -> SGDRegressor
       -> Decision trees regressor
       -> Gradient boosting regressor
       -> Random forests regressor
       Save the model, hyperparameters, and metrics in a folder named after the model class.
    """
    # store hyperparameters and model class from sgdr_hyperparameters() function call
    hyper_params, model_class = sgdr_hyperparameters()
    # store best hyperparameters after tunning 
    best_model_param = tune_regression_model_hyperparameters(model_class, hyper_params, validation=2)   
    sgdr = SGDRegressor(**best_model_param)
    # train model
    trained_model = sgdr.fit(xtrain, ytrain)
    # store score in model_score
    model_score = metrics_trained_model(trained_model)
    # save model best parameters and scores in respective folder
    save_model(trained_model, best_model_param, model_score, folder='models/regression/linear_regression')
    
    # store hyperparameters and model class from dtr_hyperparameters() function call
    hyper_params, model_class = dtr_hyperparameters()
    # store best hyperparameters after tunning
    best_model_param = tune_regression_model_hyperparameters(model_class, hyper_params, validation=2)  
    reg_decision_model=DecisionTreeRegressor(**best_model_param)
    # train model
    trained_model = reg_decision_model.fit(xtrain, ytrain)
    # store score in model_score
    model_score = metrics_trained_model(trained_model)
    # save model best parameters and scores in respective folder
    save_model(trained_model, best_model_param, model_score, folder='models/regression/decision_tree')
    
    # store hyperparameters and model class from gbr_hyperparameters() function call
    hyper_params, model_class = gbr_hyperparameters()
    # store best hyperparameters after tunning
    best_model_param = tune_regression_model_hyperparameters(model_class, hyper_params, validation=2)  
    gbr=GradientBoostingRegressor(**best_model_param)
    # train model
    trained_model = gbr.fit(xtrain, ytrain)
    # store score in model_score
    model_score = metrics_trained_model(trained_model)
    # save model best parameters and scores in respective folder
    save_model(trained_model, best_model_param, model_score, folder='models/regression/gradient_boosting')
    
    # store hyperparameters and model class from rfr_hyperparameters() function call
    hyper_params, model_class = rfr_hyperparameters()
    # store best hyperparameters after tunning
    best_model_param = tune_regression_model_hyperparameters(model_class, hyper_params, validation=2)  
    rfr=RandomForestRegressor(**best_model_param)
    # train model
    trained_model = rfr.fit(xtrain, ytrain)
    # store score in model_score
    model_score = metrics_trained_model(trained_model)
    # save model best parameters and scores in respective folder
    save_model(trained_model, best_model_param, model_score, folder='models/regression/random_forests')
    
    return

# Milestone-4, Task-7
def find_best_model() -> tuple[abc.ABCMeta, dict, dict]:
    """This function evaluates which model is best, then returns:
       -> the loaded model.
       -> a dictionary of its hyperparameters.
       -> a dictionary of its performance metrics.

    Returns:
        model (abc.ABCMeta): the model class.
        performance_metrics (dict): return a dictionary of its performance metrics.
        hyperparameters (dict): return a dictionary of its hyperparameter values.
    """
    list_of_json_df = [] # an empty list to store the data frames
    path_of_the_directory = Path('models/regression')
    for filename in os.listdir(path_of_the_directory):
        f = os.path.join(path_of_the_directory,filename)
        if os.path.isfile(f):
            pass
        else:
            json_path = os.path.join(f,'metrics.json')
            with open(json_path) as json_file:
                data = pd.read_json(json_file) # read data frame from json file
                data['Path'] = f # append the path to the data frame
                list_of_json_df.append(data) # append the data frame to the list      
    # concatenate all the data frames in the list.
    df_of_reg_metrics = pd.concat(list_of_json_df, ignore_index=True) 
    # Select the path of the directory for the model with best metrics
    best_model_path = df_of_reg_metrics[df_of_reg_metrics.validation_RMSE==df_of_reg_metrics.validation_RMSE.min()].Path.values[0]
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
    evaluate_all_models()
    best_model = find_best_model()