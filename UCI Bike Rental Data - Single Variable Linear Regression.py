#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as mplt
from math import ceil, floor


day = pd.read_csv("C:/Users/connz/Desktop/to_be_shared/Dataset/Bike Sharing/day.csv")

# 展示day.csv的前5条数据
day.head()


# 从day.csv抽取变量cnt和temp的列
temp = np.array(day['temp'])
cnt = np.array(day['cnt'])


# 用day.csv的数据绘制散点图
mplt.figure(figsize = (10, 6))
mplt.scatter(temp, cnt, s = 20, c = 'dodgerblue')
mplt.xlabel("Normalized Temperature in Celsius")
mplt.ylabel("Count of Rental Bikes")
mplt.title("Number of Rental Bikes vs. Temperature on Daily Basis")
mplt.show()


# Normal probability density function
def normfun(x, mu, sigma):
    pdf = np.exp(- ((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    return pdf


# Plot the relative frequency histogram of the variate cnt
x = np.linspace(0, 10000, 10000)
mplt.figure(figsize = (10, 6))
mplt.hist(cnt, density = True, edgecolor = 'black')
mplt.plot(x, normfun(x, np.mean(cnt), np.std(cnt)), c = 'red')
mplt.xlabel("Count of Rental Bikes")
mplt.ylabel("Relative Frequency")
mplt.title("Relative Frequency Histogram of the variate cnt")
mplt.show()


# The loss function
def loss(x, y, alpha, beta):
    return np.sum((y - (alpha + (beta * x))) ** 2)


# betahat(x, y) consumes an explanatory variate x and 
#   a response variate y and calculates the least squares
#   estimate of the regression parameter beta
# betahat: np.array np.array => numbers
# requires: x, y have the same length
#           elements in x and y are numbers
def betahat(x, y):
    n = len(x)
    assert n == len(y), "x and y have different lengths"
    numerator = np.dot(x, y) - (n * np.mean(x) * np.mean(y))
    denominator = np.dot(x, x) - (n * (np.mean(x) ** 2))
    return numerator / denominator
    
    
# alphahat(x, y) consumes an explanatory variate x and
#  a response variate y and calculates the least squares
#  estimate of the regression parameter alpha
# alphahat: np.array np.array => numbers
# requires: x, y have the same length
#           elements in x and y are numbers
def alphahat(x, y):
    n = len(x)
    assert n == len(y), "x and y have different lengths"
    return np.mean(y) - (betahat(x, y) * np.mean(x))


# Calcualte alpha^hat and beta^hat
slope = betahat(temp, cnt)
intercept = alphahat(temp, cnt)
print("The least squares estimate of the slope is", slope, "\n")
print("The least squares estimate of the intercept is", intercept, "\n")


# Plot the scatter plot with the fitted model imposed
t = np.linspace(0, 1, 100)
mplt.figure(figsize = (10, 6))
mplt.scatter(temp, cnt, s = 20, c = 'dodgerblue')
mplt.plot(t, alphahat(temp, cnt) + (betahat(temp, cnt) * t), linestyle = '--', c = 'red')
mplt.xlabel("Normalized Temperature in Celsius")
mplt.ylabel("Count of Rental Bikes")
mplt.title("Number of Rental Bikes vs. Temperature on Daily Basis")
mplt.show()


# fitted(x, y) consumes an explanatory variate x and
#  a response variate y and calculates the fitted values
# fitted: np.array np.array => np.array
# requires: x, y have the same length
#           elements in x and y are numbers
def fitted(x, y):
    return (betahat(x, y) * x) + alphahat(x, y)
    

# residual(x, y) consumes an explanatory variate x and
#  a response variate y and calculates the residuals
# residual: np.array np.array => np.array
# requires: x, y have the same length
#           elements in x and y are numbers
def residual(x, y):
    return y - fitted(x, y)


# Plot the fitted values versus the residuals
f = np.linspace(1000, 8000, 100000)
mplt.figure(figsize = (10, 6))
mplt.scatter(fitted(temp, cnt), residual(temp, cnt), color = 'white', edgecolor = "black")
mplt.plot(f, np.zeros(len(f)), c = 'red', linestyle = '--')
mplt.xlabel("Fitted Value")
mplt.ylabel("Residual")
mplt.title("Fitted Valud vs. Residual")
mplt.show()


# Define class SimpleLinearRegression
class SimpleLinearRegression:
    
    def __init__(self, csv_data, x, y, x_max = None, x_min = None, show = True, colour = 'dodgerblue', describe = True):
        data = csv_data
        title = "%s vs. %s"%(x, y)
        explanatory = np.array(data['%s'%x])
        response = np.array(data['%s'%y])
        if show:
            self.plot_model(explanatory, response, x, y, x_max, x_min, title, colour)
        if describe:
            print("Linear Model:\n\t x = %s, y = %s"%(x, y), "\n")
            print("Intercept = ", alphahat(explanatory, response))
            print("Slope = ", betahat(explanatory, response), "\n")
            
    def betahat(self, x, y):
        n = len(x)
        assert n == len(y), "x and y have different lengths"
        numerator = np.dot(x, y) - (n * np.mean(x) * np.mean(y))
        denominator = np.dot(x, x) - (n * (np.mean(x) ** 2))
        return numerator / denominator

    def alphahat(self, x, y):
        n = len(x)
        assert n == len(y), "x and y have different lengths"
        return np.mean(y) - (betahat(x, y) * np.mean(x))
        
    def plot_model(self, x, y, xlabel, ylabel, x_max, x_min, title, colour):
        left_end = x_min if x_min != None else floor(x.min())
        right_end = x_max if x_max != None else ceil(x.max())
        t = np.linspace(left_end, right_end, 100)
        mplt.figure(figsize = (10, 6))
        mplt.scatter(x, y, s = 20, c = colour)
        mplt.plot(t, alphahat(x, y) + (betahat(x, y) * t), linestyle = '--', c = 'red')
        mplt.xlabel("%s"%xlabel)
        mplt.ylabel("%s"%ylabel)
        mplt.title(title)
        mplt.show()

        

if __name__ == '__main__':
    SimpleLinearRegression(day, 'temp', 'cnt')
    SimpleLinearRegression(day, 'atemp', 'cnt', colour = "green")
    SimpleLinearRegression(day, 'hum', 'cnt', colour = "orange")
    SimpleLinearRegression(day, 'windspeed', 'cnt', colour = "purple")


# In[5]:




