#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols


# In[5]:


data = pd.read_csv('/Users/derya_ak/Desktop/marketing_sales_data.csv')


# In[9]:


data.head()


# In[14]:


# calculate the mean sales for each TV category

data.groupby('TV')['Sales'].mean()


# In[13]:


# calculate the mean sales for each Influencer category

data.groupby('Influencer')['Sales'].mean()


# In[15]:


# drop rows that contain missing data and update the dataframe.

data = data.dropna(axis=0)


# In[16]:


# eename all columns in data that contain a space because ols() function doesn't run 
#when variable names contain a space

data = data.rename(columns={'Social Media': 'Social_Media'})


# In[17]:


data.head()


# In[28]:


# create plot to see relationships

sns.pairplot(data)


# sales and radio has a line. that shows a positive association between the two variables.
# 

# In[29]:


# save resulting dataframe in a separate variable to prepare for regression

ols_data = data[["Radio", "Sales"]]


# In[30]:


ols_data.head(10)


# In[31]:


# write the linear regression formula

ols_formula = "Sales ~ Radio"


# In[32]:


# implement OLS

OLS = ols(formula = ols_formula, data = ols_data)


# In[33]:


# fit the model to the data

model = OLS.fit()


# In[34]:


#summary of results

model.summary()


# relationship between sales and radio promotion budget 
# in the form of y = slope * x + y-intercept?
# 
# y-intercept is 41.5326.
# slope is 8.1733.
# 
# #sales = 8.1733 * radio promotion budget + 41.5326
# 
# if the company spends 1 million dollars more for promoting on the radio, the company's sales increase by 8.1733 million dollars on average.

# In[35]:


#checking the model assumption this will help confirm your findings.

# plot the OLS data with the best fit regression line

sns.regplot(x = "Radio", y = "Sales", data = ols_data)


# In[42]:


# get the residuals from the model

residuals = model.resid

residuals


# In[37]:


# visualize the distribution of the residuals

fig = sns.histplot(residuals)
fig.set_xlabel("Residual Value")
fig.set_title("Histogram of Residuals")
plt.show()


# Residuals is approximately normal.This shows that the assumption of normality is likely met.

# In[ ]:


# Create a Q-Q plot 

sm.qqplot(residuals, line='s')
plt.title("Q-Q plot of Residuals")
plt.show()


# The points closely follow a straight diagonal line trending upward. This confirms that the normality assumption is met.

# In[43]:


#assumptions of independent observation and homoscedasticity.

#get fitted values

fitted_values = model.predict(ols_data["Radio"])


# In[44]:


#create a scatterplot of residuals against fitted values

fig = sns.scatterplot(x=fitted_values, y=residuals)
fig.axhline(0)
fig.set_xlabel("Fitted Values")
fig.set_ylabel("Residuals")
plt.show()


# Residuals appear to be randomly spaced, the homoscedasticity assumption seems to be met.

# P-value is 0.000 and smaller than the common significance level of 0.05
# There is a 95% probability that the interval [7.791, 8.555] contains the true value for the slope.
# 1 million dollar increase in radio promotion budget could generate average a 8.1733 million dollar increase in sales
