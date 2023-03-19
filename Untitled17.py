#!/usr/bin/env python
# coding: utf-8
### GRIP: The Spark Foundation 
### Data Science and Buisness Analytic Intern
### Task 1: Prediction Using Supervised ML
### Author: patil yogesh sanjay
##### In this task we have to predict the percentage score of a student baased on the no. of hours studied. The task has two variables where ythe feature is the no. of hours studied and the target value is the percentagr score. this can be solved using simple linear regression.
#### Step-1 import required libraries
#In[ ]:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#In[23]:
data=pd.read_csv("C:/Users/DELL/Desktop/task 1 dataset.csv")
data
### Step -2 Exploring the data
#In[49]:
data.shape
#In[24]:
data.describe()
#In[25]:
data.info()
### Step-3 Data Visualization
# In[26]:
data.plot(kind='scatter',x='hours',y='scores');
plt.show()
# In[27]:
data.corr(method='pearson')
# In[28]:
data.corr(method='spearman')
# In[29]:
hours=data['hours']
scores=data['scores']
# In[30]:
sns.distplot(hours)
# In[32]:
sns.distplot(scores)
### Step-4 Linear Regression
#In[33]:
x=data.iloc[:,:-1].values
y=data.iloc[:,1].values
# In[37]:
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=50)
# In[38]:
from sklearn.linear_model import LinearRegression
reg= LinearRegression()
reg.fit(x_train, y_train)
# In[39]:
m=reg.coef_
c=reg.intercept_
line=m*x+c
plt.scatter(x,y)
plt.plot(x,line);
plt.show()
# In[40]:
y_pred=reg.predict(x_test)
# In[41]:
actual_predicted=pd.DataFrame({'Target':y_test,'predicted':y_pred})
actual_predicted
# In[42]:
sns.set_style('whitegrid')
sns.distplot(np.array(y_test-y_pred))
plt.show()
# #### what would be the predicted score if student studies for 9.25 hours/day?
# In[43]:
h=9.25
s=reg.predict([[h]])
print("If student studies for {} hours per day he/she will score {}% in exam.".format(h,s))
### Step-5 Model Evolution
# In[47]:
from sklearn import metrics
from sklearn.metrics import r2_score
print('mean absolute error:',metrics.mean_absolute_error(y_test,y_pred))
print('R2 score:',r2_score(y_test,y_pred))
### Thank you
#In[ ]:




