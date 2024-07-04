#!/usr/bin/env python
# coding: utf-8

# ![image.png](attachment:image.png)

# # CS-331: Introduction to Artifical Intelligence
# 
# ## Linear Regression - 100 Marks
# 
# 

# #### Name:
# #### Roll Number:

#  ### Instructions:
# - The aim of this assignment is to familiarize you with Uni-variate Linear Regression and Multi-variate Linear Regreesion..
# - You can only use Python programming language and Jupyter Notebooks.
# - Submit a zip file containing both the notebooks (.ipynb file) and their python files (.py file). Name the zip file `X_PA1`, where X is your rollnumber.
# - Rename the notebook `rollnumber_PA1.ipynb` by replacing `rollnumber` with your rollnumber. Same naming convention for the .py file
# - Please make sure all cells have been run.
# - The code MUST be done independently. Any plagiarism or copying of work from others will not be tolerated.
# - You are only supposed to write your code in the regions specified! Do NOT change any code that has already been written for you!
# - You CANNOT use the scikit learn library to implement any function, unless mentioned otherwise. The entire implementation should be your own!

# ## Dataset Description
# 
# The `temperatures.csv` dataset is designed for bias correction of next-day maximum and minimum air temperature forecasts produced by the Local Data Assimilation and Prediction System (LDAPS) operated by the Korea Meteorological Administration. It covers summer seasons from 2013 to 2017 in Seoul, South Korea.
# 
# Dataset Summary:
# - **Feature Type:** Real
# - **Instances:** 7586
# - **Input Features:** 21 (including present-day temperature data, LDAPS model forecasts, and geographical information)
# - **Output:** Next-day maximum (Next_Tmax)
# 
# 
# We want to predict the next day temperature given the various features

# #### Libraries used

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


# #### Import Dataset

# Here you will load the `temperatures.csv` file using <b>Pandas</b> library

# In[2]:


# code here

df = pd.read_csv('temperatures.csv')


# ## Part 1 - Univariate Regression (40 Marks)
# 
# In this part, you will develop a univariate regression model using maximum temperature on present day `(Present_Tmax)` to predict the next day temperature `(Next_Tmax)`

# #### Feature Extraction:
# 
# Extract the `Present_Tmax` column as input and the `Next_Tmax` column as output from the dataset.

# In[3]:


#code here

df = pd.read_csv('temperatures.csv')
input  = np.array(df[['Present_Tmax']])
output =  np.array(df [['Next_Tmax']])
#print(input)
#print(output)


# #### Splitting the dataset
# Make a 70 30 split to divide the dataset into training and test dataset which will result in 4 variables: X_train, Y_train, X_test, Y_test

# In[4]:


# splitting of present T_max


X_train, X_test, Y_train, Y_test = train_test_split(input, output, test_size=0.3, random_state=42)





# ### Learn the parameters
# In this part, you will fit the linear regression parameters $\theta_0$ and $\theta_1$ to the given dataset.
# 
# The objective of linear regression is to minimize the cost function
# 
# $$ J(\theta) = \frac{1}{2m} \sum_{i=1}^m \left(\hat {y}^{(i)} - y^{(i)}\right)^2$$
# 
# where the hypothesis $\hat {y}^{(i)}$ is the predicted value for a given x and is given by the linear model and $m$ is the total number of datapoints
# $$ \hat {y} =  h_\theta(x) = \theta_0 + \theta_1 x$$
# 
# The parameters of your model are the $\theta_j$ values. These are
# the values you will adjust to minimize cost $J(\theta)$. One way to do this is to
# use the batch gradient descent algorithm. In batch gradient descent, each
# iteration performs the update
# 
# $$ \theta_0 = \theta_0 - \alpha \frac{1}{m} \sum_{i=1}^m \left( \hat {y}^{(i)} - y^{(i)}\right)$$
# 
# $$ \theta_1 = \theta_1 - \alpha \frac{1}{m} \sum_{i=1}^m \left( \hat {y}^{(i)} - y^{(i)}\right)x^{(i)}$$
# 
# With each step of gradient descent, your parameters $\theta_0$ and $\theta_1$ come closer to the optimal values that will achieve the lowest cost $J(\theta)$.

# In[5]:


def predict_SLR(X,theta_o, theta_1): #you will have to decide the parameters for this function
    #This function will return the predicted value of y based on the given parameters
    y_predicted = theta_o + theta_1*X  
    return y_predicted

def cost_SLR(X, Y,theta_o,theta_1): #you will have to decide the parameters for this function
    m = len(X)
    y_predicted = predict_SLR(X, theta_o, theta_1)
    cost = (1 / (2*m))*np.sum((y_predicted-Y)**2)
   # print(cost)
    return cost



def gradient_descent_SLR(X,Y,alpha,epochs):
    J = [] # this is a list of result of the cost function with each iteration
    #you will need to start with arbitrary values of theta0 and theta1. Initialise those here

    theta0 = 0.5
    theta1 = 0.5
    y_predicted = predict_SLR(X,theta0,theta1)
  #  print(y_predicted,Y)
    m = len(X)
    theta_1_prime = 0.5 
    theta_o_prime = 0.5 
   # while  theta_o_prime != 0 and theta_1_prime != 0 :
    for epoch in range(epochs):
     #   print("derivative : ", theta_o_prime, theta_1_prime)
        theta_o_prime = (1/m)* np.sum(y_predicted-Y)
        theta_1_prime = (1/m)* np.sum((y_predicted-Y)*X)

        #this is the loop that will implement gradient descent. With each iteration, you will need to update the values of theta0 and theta1
        #you will also have to calculate the loss with each updated value of theta0 and theta1
        # Update the parameters theta0 and theta1
        theta0 = theta0-alpha*theta_o_prime
        theta1 = theta1-alpha *theta_1_prime 
        y_predicted = predict_SLR(X,theta0,theta1)
       # print(theta0 , theta1)

        # Compute the cost and append it to the list
        cost = cost_SLR(X,Y,theta0,theta1)
        J.append(cost)

    return theta0,theta1,J


# #### Running Linear Regression
# 
# Tune the hyperparameters (epochs and learning rate) to get the best fit model for linear regression

# In[6]:


#you will have to set these values to get the best fit model for linear regression

n_epoch = 10000
alpha = 0.001

##########################
# Present T_max          #
##########################
#this is the call to the gradient descent function
#here X_train and Y_train are the arrays that you formed above using the dataset
theta0, theta1, J = gradient_descent_SLR(X_train, Y_train, alpha, n_epoch)
print('Predicted theta0 = %.4f, theta1 = %.4f, cost = %.4f' % (theta0, theta1, J[-1]))


# #### Plotting results

#  This should output a graph with the original dataset points and the linear regression model using your learned values of theta0 and theta1
# 

# In[7]:


#DO NOT EDIT THIS CELL.
y_pred_list_train = list()
for x in X_train:
    y_pred_list_train.append(predict_SLR(x, theta0, theta1))

plt.plot(X_train, Y_train, 'bo', ms=10, mec='k')
plt.ylabel('Next_Tmax')
plt.xlabel('Present_Tmax')
plt.plot(X_train, y_pred_list_train, '-')
plt.legend(['Training data', 'Linear regression'])
plt.show()

y_pred_list_test = list()
for x in X_test:
    y_pred_list_test.append(predict_SLR(x, theta0, theta1))

plt.plot(X_test, Y_test, 'ro', ms=10, mec='k')
plt.ylabel('Next_Tmax')
plt.xlabel('Present_Tmax')
plt.plot(X_test, y_pred_list_test, '-')
plt.legend(['Test data', 'Linear regression'])
plt.show()


# ### Finding the Correlation
# Correlation is used to assess the association between features (input variables) and the target variable (output variable) in a dataset.
# 
# * A positive correlation indicates a direct, linear relationship: as one variable increases, the other tends to increase as well.
# * A negative correlation indicates an inverse, linear relationship: as one variable increases, the other tends to decrease.
# * A correlation of 0 suggests no linear relationship between the variables.
# 
# <br>
# Pick the top 5 features that have the best correlation with Next_Tmax and write them in the markdown block below the code block
# We will be using the following function to plot the correlation
# 
# 
# 
# 
# 

# In[8]:


#the features parameter is all the 22 columns other than the output feature in the dataset
features = pd.read_csv('temperatures.csv')

Y_train = features['Next_Tmax'] # to make their same type


# best five are 
'''
LDAPS_Tmax_lapse 
Present Tmax
LDAPS_Tmin_lapse
present Tmin
LDAPS_LH
'''
features.drop(columns=['Next_Tmax'], inplace=True) 
#print(features)
correlations = np.corrcoef(features, Y_train, rowvar=False)[:21, -1]
#print(abs(correlations))
column_names = df.columns[:21]
plt.figure(figsize=(18, 8))
plt.bar(range(1, 22), correlations, tick_label=column_names, color='skyblue')
plt.title('Correlation Coefficients between Features and Next_Tmax')
plt.xlabel('Features')
plt.ylabel('Correlation Coefficient')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# #### Top 5 features here:
# 
# 

# ### Improving Performance
# 
# Try to improve performance of the model you have developed, i.e further reducing the cost, by selecting some other input feature instead of `Present_Tmax`, keeping the desired output as `Next_Tmax`
# 
# * Write a comment comparing the previous and the new model i.e their features, hyperparameters
# 

# In[9]:


#code here
'''
Present Tmax
LDAPS_Tmax_lapse 
LDAPS_Tmin_lapse
LDAPS_CC2
LDAPS_CC3
'''

new_input  = np.array(df[['LDAPS_Tmax_lapse']])
new_output =  np.array(df [['Next_Tmax']]) 
n_epoch = 10000
alpha = 0.001
X_train, X_test, Y_train, Y_test = train_test_split(new_input, new_output, test_size=0.3, random_state=42)
theta0, theta1, J = gradient_descent_SLR(X_train, Y_train, alpha, n_epoch)
print('Predicted theta0 = %.4f, theta1 = %.4f, cost = %.4f' % (theta0, theta1, J[-1]))
#DO NOT EDIT THIS CELL.
y_pred_list_train = list()
for x in X_train:
    y_pred_list_train.append(predict_SLR(x, theta0, theta1))

plt.plot(X_train, Y_train, 'bo', ms=10, mec='k')
plt.ylabel('Next_Tmax')
plt.xlabel('Present_Tmax')
plt.plot(X_train, y_pred_list_train, '-')
plt.legend(['Training data', 'Linear regression'])
plt.show()

y_pred_list_test = list()
for x in X_test:
    y_pred_list_test.append(predict_SLR(x, theta0, theta1))

plt.plot(X_test, Y_test, 'ro', ms=10, mec='k')
plt.ylabel('Next_Tmax')
plt.xlabel('Present_Tmax')
plt.plot(X_test, y_pred_list_test, '-')
plt.legend(['Test data', 'Linear regression'])
plt.show()


##########################
# LDAPS_Tmax_lapse       #
##########################

'''

error is reduced 
points are closer to line  
a good model because its corealtion value is 0.83 and corelation value of T_max is 0.61
''' 


# ## Part 2: Multi-Variate Linear Regression (60 Marks)
# 
# We will now use a similar concept as in the previous part to train a multivariate regression model on the same dataset. Instead of using just one input feature, we will now use the `Top-5 input features` that you have selected in the previous part. These features will be used to predict the next day temperature `(Next_Tmax)`

# #### Feature Extraction:
# 
# Extract the Top-5 features from the dataset

# In[10]:


#code here
df = pd.read_csv('temperatures.csv')
required_columns = ['Present_Tmax' ,'LDAPS_Tmax_lapse' ,'LDAPS_Tmin_lapse' ,'LDAPS_CC2' ,'LDAPS_CC3']
features = np.array((df[required_columns]))


##########################
# 5 features             #
##########################


# #### Splitting the dataset
# Make a `70 30` split to divide the dataset into training and test dataset which will result in 4 variables: X_train, Y_train, X_test, Y_test

# In[11]:


#code here  
##########################
# 5 features             #
##########################

X_train, X_test, Y_train, Y_test = train_test_split(features, new_output, test_size=0.3, random_state=42)
#print(X_train)


# ## Learn the parameters for Multivariate Regression
# 
# In multivariate regression, we predict the output using multiple input features. The model has the form:
# 
# $$
# \hat{y} = \theta_0 + \theta_1x_1 + \theta_2x_2 + ... + \theta_nx_n
# $$
# 
# where:
# - $\hat{y}$ is the predicted value
# - $x_i$ represents each input feature
# - $\theta_i$ are the parameters of our model
# 
# The cost function for multivariate regression is an extension of the univariate case and is given by:
# 
# $$
# J(\Theta) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2
# $$
# 
# Here, $\Theta$ represents the parameter vector $(\theta_0, \theta_1, ..., \theta_n)$, and $m$ is the number of training examples.
# 
# To minimize the cost function $J(\Theta)$, we use a method such as gradient descent, where each parameter $\theta_j$ is updated as follows:
# 
# $$
# \theta_j := \theta_j - \alpha \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})x_j^{(i)}
# $$
# 
# - $\alpha$ is the learning rate
# - The summation is over all training examples
# 
# 
# With each iteration of gradient descent, the parameters $\Theta$ come closer to the optimal values that minimize the cost function $J(\Theta)$.

# #### Linear Regression:
# 
# You will implement <b>Gradient Descent</b> to train the model on the data. You will be using Mean Square Error as the cost function

# In[12]:


#You can add any helper functions that you may feel the need for
def modification (input) :



 
   # std_devs = np.std(input, axis=0)
   # features_zscore = (input - mean_values) / std_devs 
    mean_values = np.mean(input, axis=0)
    max_values = np.max(input, axis=0)
    min_values = np.min(input, axis=0)

    
    normalized_input = (input - mean_values) / (max_values - min_values)

    ones_column = np.ones((normalized_input.shape[0], 1))
    X_with_bias = np.concatenate((ones_column,normalized_input ), axis=1)
    return X_with_bias

def predict_MLR(X, thetas): #decide on the variables for this function
    #this function should return the predicted values based on the value of thetas
    y_pred = np.dot(X, thetas) 
    return y_pred
  

def cost_MLR(X, Y, thetas):#decide on the parameters for this function
    #this function should return the value of the loss function
    m = len(Y)
    y_pred = predict_MLR(X, thetas)
    cost = (1 / (2 * m)) * np.sum((y_pred - Y) ** 2)
    return cost
   

#You will need to use a bias term as well
def gradient_descent_MLR(X,Y,alpha,epochs):
    row , column =  X.shape
    theta = np.full((column+1, 1), 0.5) # 6 by 1
    X_with_bias = modification(X)
  #  print(theta)
    y_predicted = predict_MLR(X_with_bias, theta)
    m = len(X) 
    theta_prime = np.full((column, 1), 0.5)
    J = []  
   
    #here you will need to initialise the values of thetas. Be mindful of the size you use for the thetas
    #thetas = ?
    for i in range(epochs):
        y_predicted = predict_MLR(X_with_bias, theta)
    
        theta_prime = (1/m)* ( np.dot(X_with_bias.T  , (np.dot(X_with_bias, theta) - Y)))
        theta =  theta - alpha *  theta_prime
       
        cost = cost_MLR(X_with_bias,Y,theta)
        J.append(cost)
       # print("cost",cost)
       # print("theeta prime",theta_prime)

    return theta, J


# #### Run the Regression
# 
# For the specified number of epochs, run the regression and store the costs and corresponding thetas for each epoch
# 

# In[13]:


#decide on an appropriate value of n_epoch and alpha
# Return J for every epoch and Plot it against # of epochs.
n_epoch = 1000
learning_rate = 0.01
thetas, J= gradient_descent_MLR(X_train,Y_train,learning_rate, n_epoch)#this function needs two more parameters. You will need to add those based on your normalised data
# print("input",X_train.shape)
# print("output",Y_train.shape )  
##########################
# 5 features             #
##########################


# ### Visualizing the costs
# You can run the following cell to see how your costs change with each epoch.

# In[14]:


plt.plot(np.arange(1,n_epoch),J[1:])
plt.xlabel("number of epochs")
plt.ylabel("cost")


# ### All input features
# 
# Now, you will use `ALL` input features to predict `Next_Tmax`.
# 
# Extract the features and run the following code block again. Make note of the differences between this model and the one that only used `5 features`.
# 

# In[15]:


#features =
features = pd.read_csv('temperatures.csv')

features.drop(columns=['Next_Tmax'],inplace=True)
#print(features)
input_1 =  np.array(features)
#print("output",new_output.shape)
#print("input",input_1.shape)

#output = 

##########################
# 22 features            #
##########################


# Perform 70-30 split on the dataset again.

# In[16]:


#your code here
X1_train, X1_test, Y1_train, Y1_test = train_test_split(input_1,new_output, test_size=0.3, random_state=42)
#print("input" ,X1_train  )
#print("output",Y1_train)


# Call the gradient descent function again

# In[17]:


#your code here
n_epoch = 1000
learning_rate = 0.01
#print("input",X1_train.shape)
#print("output",Y1_train.shape )
thetas, J1= gradient_descent_MLR(X1_train,Y1_train,learning_rate, n_epoch)
#this function needs two more parameters. You will need to add those based on your normalised data


# Visualise the cost again.

# In[18]:


#your code here


# print(J1[-1])
#print(J[-1])
plt.plot(np.arange(1,n_epoch),J1[1:])
plt.xlabel("number of epochs")
plt.ylabel("cost")


# Plot the final cost values for the previous model (with 5 features) and the new model (with all features)
# <br>
# 
# (Hint: your costs are stored in a list so you would need to extract just the last one for both the models)

# In[19]:


#your plot here
cost_final_5_features = J[-1]  
cost_final_all_features = J1[-1] 

# Plotting
models = ['5 Features Model', 'All Features Model']
final_costs = [cost_final_5_features, cost_final_all_features]
#print(final_costs)

plt.bar(models, final_costs, color=['blue', 'green'])
plt.xlabel('Models')
plt.ylabel('Final Cost')
plt.title('Final Cost Comparison')
plt.show()


# ### Analysis
# Choose the best model out of the 2 you have developed and conduct the following analysis:
# 
# * Change the learning rates to `0.01`, `0.1`, and `1` and record the performance in terms of convergence plots ie. cost vs iterations for each learning rate
# 
# * Change the train-test split ratios to `60-40`, `70-30` and `80-20` and record the performance in terms of convergence plots for each ratio i.e cost vs iterations for each split ratio
# 
# * Change the number of iterations to `250`, `500`, `750` and `1000` and record the performance in terms of convergence plots for a range of iterations i.e cost vs range of iterations
# 
# 

# #### Effect of learning rate
# Use 70-30 split and number of iterations = 1000

# In[20]:


#your code here 
X_new_train, X_new__test, Y_new__train, Y_new__test = train_test_split(input_1,new_output, test_size=0.3, random_state=42)


# Plot the `final` value of the cost against the learning rates

# In[21]:


learning_rate_list = [0.01, 0.1, 1]
cost_list = []  #you are to add the LAST value of the cost from your model to this list for each learning rate used 
n_epoch_1000 = 1000
for x in learning_rate_list :
 #print(x)
 thetas, J1= gradient_descent_MLR(X_new_train,Y_new__train,x, n_epoch)
 cost_list.append(J1[-1])


#print(cost_list)
plt.bar(learning_rate_list, cost_list, color=['blue','green','yellow'],width=0.05)
plt.title('Effect of learning rate on cost')
plt.xlabel('Learning rates')
plt.ylabel('Cost') 
'''
 - The cost values for the different learning rates are: [1.6287, 1.1549, 1.0771] for [0.01, 0.1, 1.0] learning rates respectively.
 - A learning rate of 1.0 gives the lowest cost, which might indicate that the model converges faster with a higher learning rate. However,
   care should be taken as very high learning rates can also lead to divergence or overshooting the minimum . Also fast leaarning can result in zig zag jumps 
   so a small learning rate is suggested 
'''



# #### Effect of test-train split
# Use learning rate = 0.1 and iterations = 1000

# In[22]:


#your code here  
rate_01 = 0.1 
n_epoch_1000 = 1000
split_list_trin = [0.4,0.3,0.2]
cost_list_new = []
  #you are to add the LAST value of the cost from your model to this list for each split used
for x in split_list_trin  : 
 
 X_ratio_train, X_ratio__test, Y_ratio__train, Y_ratio__test = train_test_split(input_1,new_output, test_size=x, random_state=42)
 thetas, J1= gradient_descent_MLR(X_ratio_train,Y_ratio__train,rate_01, n_epoch)
 cost_list_new.append(J1[-1])
 




# Plot the `final` value of the cost received for each split.

# In[23]:


split_list = ['60-40', '70-30', '80-20']
# print(cost_list_new)
print(cost_list_new)
plt.bar(split_list, cost_list_new, color=['blue','green','yellow'])
plt.title('Effect of test-train split on cost')
plt.xlabel('Split')
plt.ylabel('Cost') 
 '''
 The cost values for the different train-test splits are: [1.1692, 1.1549, 1.1615] for [60-40, 70-30, 80-20] splits respectively.
 With a higher proportion of data in the training set, the model tends to perform better on the test set.
However, excessively small test sets may lead to overfitting, while excessively small training sets may lead to underfitting.

 '''
 


# #### Effect of number of iterations
# 
# Use 70-30 split and learning rate = 0.1

# In[ ]:


#your code here 
iter_list = [250, 500, 750, 1000] 
cost_list_iter = []  #you are to add the LAST value of the cost from your model to this list for each iteration used

X_train, X_test, Y_train, Y_test = train_test_split(input_1,new_output, test_size=x, random_state=42) 
for x in iter_list : 
 
 
 thetas, J1= gradient_descent_MLR(X_train,Y_train,0.1, x)
 cost_list_iter.append(J1[-1])



# Plot the `final` value of the cost against each iteration.

# In[ ]:


iter_list = [250, 500, 750, 1000] 
cost_list = []  #you are to add the LAST value of the cost from your model to this list for each iteration used
print(cost_list_iter)
plt.bar(iter_list, cost_list_iter, color=['blue','green','yellow','pink'],width= 50)
plt.title('Effect of number of iterations on cost')
plt.xlabel('Number of Iterations')
plt.ylabel('Cost') 

''''
    
Increasing the number of iterations beyond a certain point may not lead to significant improvements in the cost function.
However, insufficient iterations may result in the model not converging to the optimal solution.
'''


# #### Document the effect of each change and give a concise explanation about it.

# 
