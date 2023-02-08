# modelling-airbnbs-property-listing-dataset-

## Project Documentation
In this project we will use 'listing.csv'. This dataset has 988 rows and 20 columns. 

### Milestone 3:
In this milestone we have to load the raw data in using pandas, clean the dataset and save the processed data as clean_tabular_data.csv.

**Process for cleaning the data:**
> To deal with missing values in the rating columns:
created function remove_rows_with_missing_ratings(), which removes the rows with missing values in these columns. 

> To deal with Description column: created function combine_description_strings(), which combines the list items into the same string.
It will parse the string into a list, remove empty quotes, remove any records with a missing description and remove "About this space" prefix from every description.

> To deal with empty values for some rows(i.e "guests", "beds", "bathrooms", and "bedrooms"): create a function set_default_feature_values(), which set the null values to "1". Dropped 'Unnamed: 19' column from dataset which contains nan values.

> And returned a cleaned dataset named 'clean_tabular_data.csv'.

**Process for cleaning the image data:**
> Created a function called open_image(), which takes the path of the images and open as numpy array.

> Created a function called resize_images() which loads each image, and resizes it to the same height and width before saving the new version in our new processed_images folder. It will check that every image is in RGB format, set the height of the smallest image as the height for all of the other images then set the height of the smallest image as the height for all of the other images.

> Created a function save_images(), which save the processed image in the data/processed_image folder.

Our final task in this milestone is to create a function called load_airbnb(), which returns the features and labels of our data in a tuple like (features, labels). The features will include: number of beds, number of bedrooms, number of bathrooms, number of guests, Cleanliness_rating, Accuracy_rating, Communication_rating, Location_rating, Check-in_rating, Value_rating, amenities_count and the target variable is the nightly price for the Airbnb.

### Milestone 4:
> In this milestone created a simple regression model using sklearn to train a linear regression model to predict the "Price_Night" feature from the tabular data use the built in model class SGDRegressor provided by sklearn.

> Created get_split_data(), which is used to split the data i.e 'features' into train+val and test set based on given task i.e(regression/classification).

> Evaluated the regression model performance and which produced a root mean squared error of 133.073 on the test dataset.

> Created a function called custom_tune_regression_model_hyperparameters(), which performs custom grid search by hyperparameters tuning and calculates MSE, RMSE and MAE of train, test and validation data and return the best model, a dictionary of its best hyperparameter values, and a dictionary of its performance metrics.

> Created a function called tune_regression_model_hyperparameters() which uses SKLearn's GridSearchCV perform a grid search using SKLearn's GridSearchCV over a reasonable range of hyperparameter values, train regression model using best hyper params obtained from GridSearchCV() and get best model performance metrics.

> Created save_model(), which saves the model in a file called model.joblib (SKLearn regression), its hyperparameters in a file called hyperparameters.json, its performance metrics in a file called metrics.json once it's trained and tuned.

> To improve the performance of the model by using different models provided by sklearn. For e.g., linear regression, decision trees, random forests, and gradient boosting will use the regression versions of each of these models - SGDRegressor(), DecisionTreeRegressor(), RandomForestRegressor() and GradientBoostingRegressor().

> Apply tune_regression_model_hyperparameters() to each of these models to tune their hyperparameters before evaluating them.

> Created regression_trained_model_metrics() to caculate the performance metrics of the trained model(trained and tuned using GridSearchCV) and append it to the model_score dictionary. Used this function in tune_regression_model_hyperparameters() to calculate model score.

**All Regression Models:**
> Created a function evaluate_all_models(), which evaluate all the performance of the model(regressor and/or classifier) by using different models provided by sklearn:

>SGDRegressor()
>GradientBoostingRegressor()
>RandomForestRegressor()
>DecisionTreeRegressor()

> Created a function find_best_model(), which evaluates which model is best based on validation_rmse(for regression)/accuracy(for classification) & returns the loaded model, a dictionary of its hyperparameters and dictionary of its performance metrics. Save it to the given folder name.

> Call all these created function inside if __name__ == "__main__" block.
```python

features,label = load_airbnb(dataset, label='Price_Night', numeric=True)
task_folder='models/regression'
split_data = get_split_data(features, label,task_folder='models/regression')
evaluate_all_models(task_folder, split_data)
best_reg_model = find_best_model(task_folder)

```

### Milestone 5:
> In this milestone created a simple classification model using sklearn to train a logistic regression model to predict the "category" from the tabular data using the built in model class LogisticRegression provided by sklearn.

> Adapted get_split_data(), which is used to split the data i.e 'features' into train+val and test set based on given task i.e(regression/classification).

> Evaluated the classification model performance and which produced a score of 0.4156 on the test dataset and f1_score - 0.415, precision_score - 0.416, recall_score - 0.415, accuracy_score - 69.

> Created a function called tune_classification_model_hyperparameters(), which performs grid search using SKLearn's GridSearchCV over a reasonable range of hyperparameter values, train classification model using best hyper params obtained from GridSearchCV() and get best model performance metrics.

> Adapt save_model(), which saves the model in a file called model.joblib (SKLearn regression & classification model), its hyperparameters in a file called hyperparameters.json, its performance metrics in a file called metrics.json once it's trained and tuned. The function take in the name of a folder where these files are saved as a keyword argument "folder".

> To improve the performance of the model by using different models provided by sklearn. For e.g., logistic regression, decision tree, random forest and gradient boosting will use the regression versions of each of these models - LogisticRegression(), DecisionTreeClassifier() and GradientBoostingClassifier(), RandomForestClassifier().

> Apply tune_classification_model_hyperparameters() to each of these models to tune their hyperparameters before evaluating them.

> Created classification_trained_model_metrics() to caculate the performance metrics of the trained model(trained and tuned using GridSearchCV) and append it to the model_score dictionary. Used this function in tune_classification_model_hyperparameters() to calculate model score.

**All Classification Models:**
> Created a function evaluate_all_models(), which evaluate all the performance of the model(regressor and/or classifier) by using different models provided by sklearn:

>LogisticRegression()
>GradientBoostingClassifier()
>RandomForestClassifier()
>DecisionTreeClassifier()

> Created a function find_best_model(), which evaluates which model is best based on validation_rmse(for regression)/accuracy(for classification) & returns the loaded model, a dictionary of its hyperparameters and dictionary of its performance metrics. Save it to the given folder name.

> Call all these created function inside if __name__ == "__main__" block.
```python

features,label = load_airbnb(dataset, label='Category', numeric=True)
    task_folder='models/classification'
    split_data = get_split_data(features, label,task_folder='models/classification')
    evaluate_all_models(task_folder, split_data)
    best_clf_model = find_best_model(task_folder)

```
### Milestone 6:
**Regression neural network**

In this milestone we will create a configurable neural network.

> Created a PyTorch Dataset called AirbnbNightlyPriceImageDataset that returns a tuple of (features, label) when indexed, created class AirbnbNightlyPriceImageDataset, this class represents a custom dataset for PyTorch and is used to create a custom dataset for binary or multi-class classification tasks. The dataset is constructed by passing a dataframe containing the feature data and a series containing the labels. Features should be a tensor of the numerical tabular features of the house. The second element is a scalar of the price per night.

> Create a PyTorch Dataset called AirbnbNightlyPriceImageDataset which returns a tuple of (features, label). The features should be a tensor of the numerical tabular features of the house. The second element is a scalar of the price per night. Create a dataloader for the train set and test set. Split the train set into train and validation.

> This pytorch dataset should only ingest the numerical tabular data. Defined a function called train() which takes in the model, the data loader, and the number of epochs. Get the first batch of data from the dataloader and pass it through the model, then break out of the training loop. This function is used to Train a neural network for either regression or classification.

> Created the training loop and train the model to completion, Complete the training loop so that it iterates through every batch(i.e for train_set batch=64, test_set = 32, val_set=64) in the dataset for the specified number of epochs(which is 200), and optimises(optimisers are 'sgd', 'adam', 'adagrad') the model parameters.

> Used tensorboard to visualize the training curves of the model and the accuracy both on the training and validation set.

> Created a YAML file called nn_config.yaml, next to modelling.py file, that defines the architecture of the neural network which specify:-

    -The name of the optimiser used under a key called optimiser
    -The learning rate
    -The width of each hidden layer under a key called hidden_layer_width (For simplicity, all of the hidden layers have the same width)
    -The depth of the model

> Defined a function called get_nn_config() which returns the configuration for a neural network model as a dictionary, loaded from a YAML file.

> We will pass the config as a parameter to train() function as the hyperparameter dictionary, which is defined earlier.

> Specify a keyword argument called "config" which must be passed to model class upon initialisation.

> Network should then use that config to set the corresponding hyperparameters.

> Created a new folder named neural_networks, Inside it, created a folder called regression.

> Adapted save_model() function , it detects whether the model is PyTorch, and if so then:

    -saves the torch model in a file called model.pt
    -its hyperparameters in a file called hyperparameters.json
    -its performance metrics in a file called metrics.json.

> Metrics should include:-

    -The RMSE loss of the model under a key called RMSE_loss for training, validation, and test sets.
    -The R^2 score of your model under a key called R_squared for training, validation, and test sets.
    -The time taken to train the model under a key called training_duration.
    -The average time taken to make a prediction under a key called inference_latency.

> Created a class NNRegressionModel for building a neural network for regression tasks.

> Computes regression evaluation metrics for a PyTorch neural network by creating a function nn_metrics_regression(), which computes the Root Mean Squared Error (RMSE) and the R-squared (R2) values for a PyTorch neural network model on a given set of input and target data.

> Every time after training a model, create a new folder whose name will be current date and time.

> For example: a model trained on the 1st of January at 08:00:00 would be saved in a folder called models/regression/neural_networks/2018-01-01_08:00:00.

> Defined a function generate_nn_configs() which will create many config dictionaries for network. This function generates a list of dictionaries of hyperparameters for a neural network model, based on a task string input of either "regression" or "classification".

> Defined a function called find_best_nn() which will call this function(generate_nn_configs()) and then sequentially trains models with each config. This function Finds the best neural network model based on the supervised learning task (i.e regression or classification)and returns the best model with performance metrics.

> Saved the best model in folder and tried different parameterisations of the network.

### Milestone 7:
**Classification neural network**

In this milestone we reused the framework for another use-case with the Airbnb data.

> Adapted load_airbnb(), which returns the features and labels of our data in a tuple like (features, labels).

> For the Airbnb dataset, the output labels are from 1 to 10 except 9, this needs to change because PyTorch supports labels starting from 0. That is [0, n]. Output labels need to be remapped so that it starts from 0, this can also be implemented using preprocessing.LabelEncoder(). So, Created reverse_mapping(), which is used to remap the output label to starts from 0.

> Created a function get_processed_data() which is used to preprocess tha data of output label for both nn_classification modelling & nn_regression modelling.

> Adapted get_split_data(), which is used to split the data i.e 'features' into train+val and test set based on given task i.e(regression/classification).

> Due to class imbalance, we use stratified split to create our train, validation, and test sets. it still does not ensure that each mini-batch of our model see’s all our classes need to over-sample the classes with less number of values. to do that, we use the WeightedRandomSampler.

    -Created weighted_sampling(), which performs weighted random sampling on a train dataset of train_dataset and its target values of y_train to balance the classes in the dataset. The output is a WeightedRandomSample object which can be used as a data sampler for PyTorch's DataLoader.

> Adapted get_dataloader(), which creates PyTorch dataloaders for train, test and validation datasets based on the task type (regression or classification). For the classification task, the dataloader is over-sampled to handle class imbalance.

> Adapted find_best_nn(), which will call this function(generate_nn_configs()) and then sequentially trains models with each config. This function Finds the best neural network model based on the supervised learning task (i.e regression or classification)and returns the best model with performance metrics and save the model in given folder location.