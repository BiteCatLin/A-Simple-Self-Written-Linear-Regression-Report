#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as mplt


day = pd.read_csv("C:/Users/connz/Desktop/to_be_shared/Dataset/Bike Sharing/day.csv")

# 展示day.csv的前5条数据
day.head()


# 从day.csv抽取相应的列
temp = np.array(day['temp'])
atemp = np.array(day['atemp'])
hum = np.array(day['hum'])
ws = np.array(day['windspeed'])
cnt = np.array(day['cnt'])


# Normal probability density function
def normfun(x, mu, sigma):
    pdf = np.exp(- ((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    return pdf

x = np.linspace(0, 10000, 10000)
mplt.figure(figsize = (10, 6))
mplt.hist(cnt, density = True, edgecolor = 'black')
mplt.plot(x, normfun(x, np.mean(cnt), np.std(cnt)), c = 'red')
mplt.xlabel("Count of Rental Bikes")
mplt.ylabel("Relative Frequency")
mplt.title("Relative Frequency Histogram of the variate cnt")
mplt.show()


# 分别绘制散点图
mplt.figure(figsize = (20, 15))

# cnt vs. temp
mplt.subplot(2, 2, 1)
mplt.scatter(temp, cnt, s = 20, c = 'dodgerblue')
mplt.xlabel("Normalized Temperature in Celsius")
mplt.ylabel("Count of Rental Bikes")
mplt.title("Number of Rental Bikes vs. Temperature on Daily Basis")

# cnt vs. atemp
mplt.subplot(2, 2, 2)
mplt.scatter(atemp, cnt, s = 20, c = 'green')
mplt.xlabel("Normalized Feeling Temperature in Celsius")
mplt.ylabel("Count of Rental Bikes")
mplt.title("Number of Rental Bikes vs. Feeling Temperature on Daily Basis")

# cnt vs. hum
mplt.subplot(2, 2, 3)
mplt.scatter(hum, cnt, s = 20, c = 'orange')
mplt.xlabel("Normalized Humidity")
mplt.ylabel("Count of Rental Bikes")
mplt.title("Number of Rental Bikes vs. Humidity on Daily Basis")

# cnt vs. windspeed
mplt.subplot(2, 2, 4)
mplt.scatter(ws, cnt, s = 20, c = 'purple')
mplt.xlabel("Normalized Windspeed")
mplt.ylabel("Count of Rental Bikes")
mplt.title("Number of Rental Bikes vs. Windspeed on Daily Basis")

mplt.show()


# 加1后取对数并再次绘制散点图
# 从day.csv抽取相应的列
log_temp = np.log(temp + 1)
log_atemp = np.log(atemp + 1)
log_hum = np.log(hum + 1)
log_ws = np.log(ws + 1)
log_cnt = np.log(cnt + 1)


mplt.figure(figsize = (20, 15))

# log_cnt vs. log_temp
mplt.subplot(2, 2, 1)
mplt.scatter(log_temp, log_cnt, s = 20, c = 'dodgerblue')
mplt.xlabel("log_temp")
mplt.ylabel("log_cnt")
mplt.title("log_cnt vs. log_temp")

# log_cnt vs. log_atemp
mplt.subplot(2, 2, 2)
mplt.scatter(log_atemp, log_cnt, s = 20, c = 'green')
mplt.xlabel("log_atemp")
mplt.ylabel("log_cnt")
mplt.title("log_cnt vs. log_atemp")

# log_cnt vs. log_hum
mplt.subplot(2, 2, 3)
mplt.scatter(log_hum, log_cnt, s = 20, c = 'orange')
mplt.xlabel("log_hum")
mplt.ylabel("log_cnt")
mplt.title("log_cnt vs. log_hum")

# log_cnt vs. log_windspeed
mplt.subplot(2, 2, 4)
mplt.scatter(log_ws, log_cnt, s = 20, c = 'purple')
mplt.xlabel("log_ws")
mplt.ylabel("log_cnt")
mplt.title("log_cnt vs. log_ws")

mplt.show()


# 绘制log_cnt的箱型图
# Plot boxplot for log_cnt
mplt.figure(figsize = (8, 8))
mplt.boxplot(log_cnt, sym = "+")
mplt.xlabel("log_cnt")
mplt.title("Boxplot of log_cnt")
mplt.show()


# 晒出异常值并重新绘制散点图
# Store log_temp, log_atemp, log_hum, log_ws, log_cnt in a new data frame
log_day = pd.DataFrame(data = np.array([log_temp, log_atemp, log_hum, log_ws, log_cnt]),                        index = ['log_temp', 'log_atemp', 'log_hum', 'log_ws', 'log_cnt']).T
print("number of row:", log_day.shape[0])
print("number of column:", log_day.shape[1])

# Find interquartile range of log_cnt
summary = log_day.describe()
lower = summary['log_cnt'].loc['25%']
upper = summary['log_cnt'].loc['75%']
iqr = upper - lower

# Filter out outliers
lower_bound = lower - (1.5 * iqr)
log_day_remove_outlier = log_day.loc[log_day['log_cnt'] > lower_bound]
print("number of row:", log_day_remove_outlier.shape[0])
print("number of column:", log_day_remove_outlier.shape[1])

# Extract each column
log_temp_remove_outlier = log_day_remove_outlier['log_temp']
log_atemp_remove_outlier = log_day_remove_outlier['log_atemp']
log_hum_remove_outlier = log_day_remove_outlier['log_hum']
log_ws_remove_outlier = log_day_remove_outlier['log_ws']
log_cnt_remove_outlier = log_day_remove_outlier['log_cnt']


mplt.figure(figsize = (20, 15))

# log_cnt vs. log_temp
mplt.subplot(2, 2, 1)
mplt.scatter(log_temp_remove_outlier, log_cnt_remove_outlier, s = 20, c = 'dodgerblue')
mplt.xlabel("log_temp")
mplt.ylabel("log_cnt")
mplt.title("log_cnt vs. log_temp with Outliers Removed")

# log_cnt vs. log_atemp
mplt.subplot(2, 2, 2)
mplt.scatter(log_atemp_remove_outlier, log_cnt_remove_outlier, s = 20, c = 'green')
mplt.xlabel("log_atemp")
mplt.ylabel("log_cnt")
mplt.title("log_cnt vs. log_atemp with Outliers Removed")

# log_cnt vs. log_hum
mplt.subplot(2, 2, 3)
mplt.scatter(log_hum_remove_outlier, log_cnt_remove_outlier, s = 20, c = 'orange')
mplt.xlabel("log_hum")
mplt.ylabel("log_cnt")
mplt.title("log_cnt vs. log_hum with Outliers Removed")

# log_cnt vs. log_windspeed
mplt.subplot(2, 2, 4)
mplt.scatter(log_ws_remove_outlier, log_cnt_remove_outlier, s = 20, c = 'purple')
mplt.xlabel("log_ws")
mplt.ylabel("log_cnt")
mplt.title("log_cnt vs. log_ws with Outliers Removed")

mplt.show()


# loss(X, y, beta) calculates the sum of squares of
#   errors in a multivariate linear regression model
# loss: np.matrix np.matrix np.matrix => Num
# requires: elements in X, y, beta are numeric
#           y and beta have the same length
#           the number of columns in X is equal to the length of beta

def loss(X, y, beta):
    return (y.T * y) - (beta.T * X.T * y) - (y.T * X * beta) + (beta.T * X.T * X * beta)


# GradientDescent(X, y, beta, rate, step) consumes a matrix X,
#   a matrix y, a matrix beta, a number rate, and a natural 
#   number steps, and performs gradient descent to find the
#   regression vector betahat
# GradientDescent: np.matrix np.matrix np.matrix Num Nat => np.array
# requires: rate > 0
#           step >= 1
#           elements in X, y, beta are numeric
#           y and beta have the same length
#           the number of columns in X is equal to the length of beta

def GradientDescent(X, y, beta, rate, step):
    
    # the last step
    if step == 0:
        return beta
    
    # calculate the next beta vector
    next_beta = beta - (2 * rate * X.T * ((X * beta) - y))
    
    # if the loss starts to increase again, stop immediately
    if loss(X, y, beta) <= loss(X, y, next_beta):
        return beta
    
    # otherwise, the recursion continues 
    else:
        return GradientDescent(X, y, next_beta, rate, step - 1)


    
# 计算线性回归向量    
# Create matrix X and X^T in this case
n = log_day_remove_outlier.shape[0]                 # 731 observed values in this case
k = 4                                               # 4 variates in this case
first_column = np.ones(n)

x_matrix_transpose = np.mat([first_column,                                np.array(log_day_remove_outlier['log_temp']),                                np.array(log_day_remove_outlier['log_atemp']),                                np.array(log_day_remove_outlier['log_hum']),                                np.array(log_day_remove_outlier['log_ws'])])

x_matrix = x_matrix_transpose.T

# Create vector y in this case
y_vector = np.mat([np.array(log_day_remove_outlier['log_cnt'])]).T
print(" log_temp, log_atemp, log_hum, log_ws 组成的矩阵的一部分:\n")
print("X = ", x_matrix)
print("\n")
print("log_cnt 的前10个观察值:\n")
print("Y = ", y_vector[0 : 10], " ...")
print("\n\n")

# Initialize beta_0
beta_0 = np.mat([10, 10, 10, 10, 10]).T
print("The loss due to beta_0 is:", loss(x_matrix, y_vector, beta_0)[0, 0], "\n")

# learning rate and steps
rate = 0.0001
step = 1000

# the resultant regression vector
beta_1000 = GradientDescent(x_matrix, y_vector,beta_0, rate, step)
print("The new beta vector is: \n\n ", beta_1000, "\n")
print("The loss due to beta_1000 is:", loss(x_matrix, y_vector, beta_1000)[0, 0], "\n")


# Plot the loss versus the number of steps in the
#   gradient descent process

s = np.linspace(1, 50, 50)
current_beta = beta_0
lst_beta = []
lst_loss = []
for step in s:
    current_beta = GradientDescent(x_matrix, y_vector, current_beta, rate, 1)
    lst_loss.append(loss(x_matrix, y_vector, current_beta)[0, 0])
        
mplt.figure(figsize = (10, 5))    
mplt.plot(s, lst_loss, linestyle = '--', c = 'red')
mplt.xlabel("Number of Steps")
mplt.ylabel("Loss")
mplt.title("Loss vs. Steps")
mplt.show()


# 用不同的学习率绘制损失与递归次数的曲线

mplt.figure(figsize = (12, 5))

# alpha = 1.0
s0 = np.linspace(1, 50, 50)
current_beta0 = beta_0
lst_beta0 = []
lst_loss0 = []
for step in s0:
    current_beta0 = GradientDescent(x_matrix, y_vector, current_beta0, 1.0, 1)
    lst_loss0.append(loss(x_matrix, y_vector, current_beta0)[0, 0])  
mplt.plot(s0, lst_loss0, linestyle = '--', c = 'purple', label="rate = 1.0")


# alpha = 0.1
s1 = np.linspace(1, 50, 50)
current_beta1 = beta_0
lst_beta1 = []
lst_loss1 = []
for step in s1:
    current_beta1 = GradientDescent(x_matrix, y_vector, current_beta1, 0.1, 1)
    lst_loss1.append(loss(x_matrix, y_vector, current_beta1)[0, 0])    
mplt.plot(s1, lst_loss1, linestyle = '--', c = 'blue', label="rate = 0.1")


# alpha = 0.01
s2 = np.linspace(1, 50, 50)
current_beta2 = beta_0
lst_beta2 = []
lst_loss2 = []
for step in s2:
    current_beta2 = GradientDescent(x_matrix, y_vector, current_beta2, 0.01, 1)
    lst_loss2.append(loss(x_matrix, y_vector, current_beta2)[0, 0])    
mplt.plot(s2, lst_loss2, linestyle = '--', c = 'orange', label="rate = 0.01")


# alpha = 0.001
s3 = np.linspace(1, 50, 50)
current_beta3 = beta_0
lst_beta3 = []
lst_loss3 = []
for step in s3:
    current_beta3 = GradientDescent(x_matrix, y_vector, current_beta3, 0.001, 1)
    lst_loss3.append(loss(x_matrix, y_vector, current_beta3)[0, 0])      
mplt.plot(s3, lst_loss3, linestyle = '--', c = 'grey', label="rate = 0.001")


# alpha = 0.0001
s4 = np.linspace(1, 50, 50)
current_beta4 = beta_0
lst_beta4 = []
lst_loss4 = []
for step in s4:
    current_beta4 = GradientDescent(x_matrix, y_vector, current_beta4, 0.0001, 1)
    lst_loss4.append(loss(x_matrix, y_vector, current_beta4)[0, 0])    
mplt.plot(s4, lst_loss4, linestyle = '--', c = 'red', label="rate = 0.0001")


# alpha = 0.00001
s5 = np.linspace(1, 50, 50)
current_beta5 = beta_0
lst_beta5 = []
lst_loss5 = []
for step in s5:
    current_bet5a = GradientDescent(x_matrix, y_vector, current_beta5, 0.00001, 1)
    lst_loss5.append(loss(x_matrix, y_vector, current_beta5)[0, 0])   
mplt.plot(s5, lst_loss5, linestyle = '--', c = 'brown', label="rate = 0.00001")


# alpha = 0.000001
s6 = np.linspace(1, 50, 50)
current_beta6 = beta_0
lst_beta6 = []
lst_loss6 = []
for step in s6:
    current_beta6 = GradientDescent(x_matrix, y_vector, current_beta6, 0.000001, 1)
    lst_loss6.append(loss(x_matrix, y_vector, current_beta6)[0, 0]) 
mplt.plot(s6, lst_loss6, linestyle = '--', c = 'cyan', label="rate = 0.000001")


# alpha = 0.0000001
s7 = np.linspace(1, 50, 50)
current_beta7 = beta_0
lst_beta7 = []
lst_loss7 = []
for step in s7:
    current_beta7 = GradientDescent(x_matrix, y_vector, current_beta7, 0.0000001, 1)
    lst_loss7.append(loss(x_matrix, y_vector, current_beta7)[0, 0])  
mplt.plot(s7, lst_loss7, linestyle = '--', c = 'magenta', label="rate = 0.0000001")

mplt.xlabel("Number of Steps")
mplt.ylabel("Loss")
mplt.title("Loss vs. Step in Gradient Descent")
mplt.legend()
mplt.show()


# Define class LinearRegression
class LinearRegression:
    
    def __init__(self, X, y, beta, rate = 0.01, show_input = True, step = 100, show_loss = True):
        assert not rate <= 0, "rate must be positive"
        assert not step <= 0, "step must be positive"
        assert X.shape[0] == y.shape[0], "X and y have different sizes"
        assert beta.shape[0] == X.shape[1], "Sizes of X and beta do not allow  matrix multiplication"
        print("Find linear model to fit y and X:", end = "")
        if show_input:
            if len(y) < 11:
                print("\n\n X = %s,\n\n y = %s"%(X, y), "\n")
            if len(y) > 10:
                print("\n\n X = %s,\n\n y = %s ..."%(X, y if len(y) < 11 else y[:10, 0]), "\n")
        print("\nFrom initial value\n beta = \n ", beta, "\n")
        print("At rate %s and perform gradient descent for %s times:\n"%(rate, step))
        betahat = GradientDescent(X, y, beta, rate, step)
        print("\n\nThe resultant value is:\n betahat = \n ", betahat, "\n")
        if show_loss:
            print("The loss at betahat is %s"%(loss(X, y, betahat)))
        
               
    def loss(self, X, y, beta):
        return ((y.T * y) - (beta.T * X.T * y) - (y.T * X * beta) + (beta.T * X.T * X * beta))[0, 0]

    
    def GradientDescent(self, X, y, beta, rate, step):
        if step == 0:
            return beta
        next_beta = beta - (2 * rate * X.T * ((X * beta) - y))
        if loss(X, y, beta) <= loss(X, y, next_beta):
            print("Next step produces greater loss, gradient descent stopped with %d steps remaining"%step)
            return beta
        else:
            return GradientDescent(X, y, next_beta, rate, step - 1)
        

if __name__ == '__main__':
    LinearRegression(x_matrix, y_vector, beta_0, rate = 0.0001, step = 1000)

