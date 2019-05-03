#!/usr/bin/env python
# coding: utf-8

# In[5]:


import csv
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt


# In[6]:


#reading the csv file by creating dates and prices array and storing into it. 
dates = []
prices = []

def get_data(filename):
    with open(filename, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader) # skipping column names
        for row in csvFileReader:
            dates.append(int(row[0]))
            prices.append(float(row[1]))
    return


# In[7]:



def predict_price(dates, prices, b):
    dates = np.reshape(dates, (len(dates),1)) # converting to matrix of n X 1
    prices = np.reshape(prices, (len(prices),1))# converting to matrix of n X 1
    b = np.array(b).reshape(-1,1) #reshaping the predicting date value. The (-1,1) in reshape basically says that the last axis should be of size 1 and the first axis should have a size that doesn't change the total size of the array.
    linear_mod = linear_model.LinearRegression() # defining the linear regression model
    linear_mod.fit(dates, prices) # fitting the data points in the model
    plt.scatter(dates, prices, color= 'black', label= 'Data') # plotting the initial datapoints 
    plt.plot(dates, linear_mod.predict(dates), color= 'red', label= 'Linear model') # plotting the line made by linear regression
    plt.xlabel('Date') #labeling the x axis
    plt.ylabel('Price') #labeling the y axis
    plt.title('Linear Regression') #labeling the title
    plt.legend() #it automatically creates a legend for any labeled plot elements
    plt.show() #viewing the plotted graph
 
    return linear_mod.predict(b)[0][0], linear_mod.coef_[0][0], linear_mod.intercept_[0] # returning the predicted price, coefficent and intercept value


# In[8]:


get_data('google.csv') # calling get_data method by passing the csv file to it
print ("Dates- ", dates) #displaying the dates value
print ("Prices- ", prices) ##displaying the price value


# In[9]:


predicted_price, coefficient, constant = predict_price(dates, prices, 29)#calling the predict_price definition and passing values
print ("\nThe stock open price for 25th May is: $",predicted_price)
print ("The regression coefficient is ", str(coefficient), ", and the constant is ", str(constant))

