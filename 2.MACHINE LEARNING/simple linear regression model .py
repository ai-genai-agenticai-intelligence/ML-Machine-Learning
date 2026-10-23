# import require library 

import numpy as np 	

import matplotlib.pyplot as plt

import pandas as pd	

# import the dataset

dataset = pd.read_csv(r'C:\Users\A3MAX SOFTWARE TECH\Desktop\WORK\1. KODI WORK\1. NARESH\2. EVENING BATCH\N_Batch -- 7.30PM\3. Sep\6th- slr\SIMPLE LINEAR REGRESSION\Salary_Data.csv')

# split the data to independent variable 
X = dataset.iloc[:, :-1].values

# split the data to dependent variabel 
y = dataset.iloc[:,1].values 

# as d.v is continus that regression algorithm 
# as in the data set we have 2 attribute we slr algo

# split the dataset to 80-20%
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

#we called simple linear regression algoriytm from sklearm framework 
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

# we build simple linear regression model regressor
regressor.fit(X_train, y_train)


# test the model & create a predicted table 
y_pred = regressor.predict(X_test)

# visualize train data point ( 24 data)
plt.scatter(X_train, y_train, color = 'red') 
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# visulaize test data point 
plt.scatter(X_test, y_test, color = 'red') 
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# slope is generrated from linear regress algorith which fit to dataset 
m = regressor.coef_

# interceppt also generatre by model. 
c = regressor.intercept_

# predict or forcast the future the data which we not trained before 
y_12 = 9312 * 12 + 26780

y_20 = 9312 * 20 + 26780


# to check overfitting  ( low bias high variance)
bias = regressor.score(X_train, y_train)
bias


# to check underfitting (high bias low variance)
variance = regressor.score(X_test,y_test)
variance


# deployment in flask & html 
# mlops (azur, googlcolab, heroku, kubarnate)

import pickle

# Save the trained model to disk
filename = 'linear_regression_model.pkl'

# Open a file in write-binary mode and dump the model
with open(filename, 'wb') as file:
    pickle.dump(regressor, file)

print("Model has been pickled and saved as linear_regression_model.pkl")

