import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from tabular_data import load_airbnb
dataset = pd.read_csv("clean_tabular_data.csv")
features,labels = load_airbnb(dataset, label='Price_Night')

# assign the features to x and labels to y
x = features
y = labels

# split the dataset into train & test data using train_test split()
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

# create an SGDClassifier instance which will have methods to do linear regression 
# fitting by gradient descent
sgdr = SGDRegressor(loss="huber", epsilon=0.2)

# train the model by fit method
sgdr.fit(xtrain, ytrain)

# predict target using xtest data
ypred_test = sgdr.predict(xtest)

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


