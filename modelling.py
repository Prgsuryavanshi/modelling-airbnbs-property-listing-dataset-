import os
import abc
import json
import joblib
import itertools
import warnings
import pandas as pd
from pathlib import Path
from sklearn.linear_model import SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

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

# Milestone-4, Task-4 & 6
def tune_regression_model_hyperparameters(model_class: abc.ABCMeta, hyper_params: dict, xtrain: pd.core.frame.DataFrame, ytrain: pd.core.series.Series, validation: int=5) -> tuple[abc.ABCMeta, dict, dict]:
    """This function uses SKLearn's GridSearchCV to perform 
        a grid search over a reasonable range of hyperparameter values.

    Args:
        model_class (abc.ABCMeta): the model class to be optimized by cross-validated grid-search over a parameter grid.
        hyper_params (dict): a dictionary of hyperparameter names mapping to a list of values to be tried.
        xtrain (pd.core.frame.DataFrame): training set.
        ytrain (pd.core.series.Series): training target set.
        validation (int, optional): the cross-validation splitting strategy. Defaults to 5.

    Returns:
        trained_model (abc.ABCMeta): the model class trained and tuned using GridSearchCV.
        best_model_param (dict): a dictionary of best hyperparameters on which the model is trained.
        model_score (dict): a dictionary of performance metrics of the model once it is trained and tuned.
    """
    grid = GridSearchCV(estimator=model_class, param_grid=hyper_params, cv=validation, scoring='r2', verbose=1, n_jobs=-1)  
    grid_result = grid.fit(xtrain, ytrain)
    best_model_param = grid_result.best_params_

    reg_model = model_class.__class__
    reg_model = reg_model(**best_model_param)
    # train model
    trained_model = reg_model.fit(xtrain, ytrain)
    # store score in model_score
    model_score = metrics_trained_model_regressor(trained_model)

    return trained_model, model_score, best_model_param

def metrics_trained_model_regressor(trained_model: abc.ABCMeta) -> dict:
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

# Milestone-5, Task-5
def metrics_trained_model_classifier(trained_model: abc.ABCMeta) -> dict:
    """This function caculates the performance metrics of the trained classifier model
    (trained and tuned using GridSearchCV) and append it to the model_score dictionary.

    Args:
        trained_model (abc.ABCMeta): the model class trained and tuned using GridSearchCV.

    Returns:
        model_score (dict): a dictionary of performance metrics.
    """
    model_score = {"validation_accuracy":[], "f1_score":[], "precision_score":[], "recall_score":[], "train_accuracy":[]}
    ypred_test = trained_model.predict(xtest)
    # predict target using xtrain data
    ypred_train = trained_model.predict(xtrain)
    # calculate f1 score with test data
    test_f1_score = f1_score(ytest, ypred_test, average='micro')
    # calculate precision score with test data
    test_precision_score = precision_score(ytest, ypred_test, average='macro')
    # calculate recall score with test data
    test_recall_score = recall_score(ytest, ypred_test, average='micro')
    test_accuracy = accuracy_score(ytest, ypred_test, normalize=True)
    train_accuracy_score = accuracy_score(ytest, ypred_test, normalize=True)
    model_score["validation_accuracy"].append(test_accuracy)
    model_score["f1_score"].append(test_f1_score)
    model_score["precision_score"].append(test_precision_score)
    model_score["recall_score"].append(test_recall_score)
    model_score["train_accuracy"].append(train_accuracy_score)

    return model_score

def tune_classification_model_hyperparameters(model_class: abc.ABCMeta, hyper_params: dict, xtrain: pd.core.frame.DataFrame, ytrain: pd.core.series.Series, validation: int=5) -> tuple[abc.ABCMeta, dict, dict]:
    """This function uses SKLearn's GridSearchCV to perform a grid search over a reasonable range of 
       hyperparameter values and evaluates the performance using a different metric:-
       -> f1_score, precision_score, recall_score, accuracy_score

    Args:
        model_class (abc.ABCMeta): the model class to be optimized by cross-validated grid-search over a parameter grid.
        hyper_params (dict): a dictionary of hyperparameter names mapping to a list of values to be tried.
        xtrain (pd.core.frame.DataFrame): training set.
        ytrain (pd.core.series.Series): training target set.
        validation (int, optional): Defaults to 5.

    Returns:
        trained_model (abc.ABCMeta): the model class trained and tuned using GridSearchCV.
        best_model_param (dict): a dictionary of best hyperparameters on which the model is trained.
        model_score (dict): a dictionary of performance metrics of the model once it is trained and tuned.
    """
    grid = GridSearchCV(estimator=model_class, param_grid=hyper_params, cv=validation, scoring='accuracy', verbose=1, n_jobs=-1)  
    grid_result = grid.fit(xtrain, ytrain)
    best_model_param = grid_result.best_params_

    clf_model = model_class.__class__
    clf_model = clf_model(**best_model_param)
    # train model
    trained_model = clf_model.fit(xtrain, ytrain)
    # store score in model_score
    model_score = metrics_trained_model_classifier(trained_model)

    return trained_model, model_score, best_model_param

# Milestone-4, Task-6
def evaluate_all_models(task_folder: str) -> None:
    """This function evaluate all the performance of the model(regressor and/or classifier) 
       by using different models provided by sklearn:
       -> SGDRegressor(Linear regressor)
       -> LogisticRegression(Linear classifier)
       -> Decision trees (regressor and classifier)
       -> Gradient boosting (regressor and classifier)
       -> Random forests (regressor and classifier)
       Save the model, hyperparameters, and metrics in a folder named after the model class.

    Args:
        task_folder (str): the name of the parent folder where evaluated regressor and classifier models
        are saved.
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
    classifier_models = {
        'logistic_regression' : [LogisticRegression(), sgd_clf_parameters],
        'decision_tree'       : [DecisionTreeClassifier(), decisiontree_clf_parameters],
        'gradient_boosting'   : [GradientBoostingClassifier(), gradientboosting_clf_parameters],
        'random_forests'      : [RandomForestClassifier(), randomforest_clf_parameters],
    }

    if task_folder == 'models/regression':
        for key in regressor_models:
            trained_model, model_score, best_model_param = tune_regression_model_hyperparameters(regressor_models[key][0],regressor_models[key][1], xtrain, ytrain, validation=2)
            save_model(trained_model, best_model_param, model_score, folder=os.path.join(task_folder,key))
    
    elif task_folder == 'models/classification':
        for key in classifier_models:
            trained_model, model_score, best_model_param = tune_classification_model_hyperparameters(classifier_models[key][0],classifier_models[key][1], xtrain, ytrain, validation=2)
            save_model(trained_model, best_model_param, model_score, folder=os.path.join(task_folder,key))
    else:
        pass

    return
# Milestone-4, Task-7
def find_best_model(task_folder: str) -> tuple[abc.ABCMeta, dict, dict]:
    """This function evaluates which model is best, then returns:
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
    list_of_json_df = [] # an empty list to store the data frames
    path_of_the_directory = Path(task_folder)
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
    if str(path_of_the_directory) == 'models/regression':
        best_model_path = df_of_reg_metrics[df_of_reg_metrics.validation_RMSE==df_of_reg_metrics.validation_RMSE.min()].Path.values[0]
    elif str(path_of_the_directory) == 'models/classification':
        best_model_path = df_of_reg_metrics[df_of_reg_metrics.validation_accuracy==df_of_reg_metrics.validation_accuracy.min()].Path.values[0]
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
    # tune, train, save and find best regression model
    task_folder='models/regression'
    evaluate_all_models(task_folder)   
    best_reg_model = find_best_model(task_folder)
    print(best_reg_model)
    # tune, train, save and find best classification model
    task_folder='models/classification'
    evaluate_all_models(task_folder)
    best_clf_model = find_best_model(task_folder)
    print(best_clf_model)