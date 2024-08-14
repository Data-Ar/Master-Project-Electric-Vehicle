#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import packages....


# In[7]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import re
import plotly.express as px
import cartopy.crs as ccrs


# In[8]:


df = pd.read_csv('us_car_data.csv')


# In[9]:


df.head()


# In[10]:


df.shape


# In[11]:


df.describe()


# In[12]:


df.info()


# In[13]:


# to check for missing values in the dataset


# In[14]:


df.isnull().sum()


# In[15]:


# from the output above, it is evident that there are no null values in our dataset.


# In[16]:


# Feature engineering
# I realised that in our dataset,we have three columns representing different levels of 
# geographical divisions: County, City, and State. 
# To create a more informative and combined representation of these geographical attributes, 
# we will create a new column called Location.


# In[17]:


# The Location column will be a concatenation of the County, City, and State columns, 
# with each value separated by a comma. 
# For example, if a row has the values "Yakima" for County, "Yakima" for City, and "WA" for State, 
# the corresponding Location value will be "Yakima, Yakima, WA".


# In[18]:


df['Location'] = df['County'] + ', ' + df['City'] + ', ' + df['State']


# In[19]:


# # Feature Engineering: Creating a Price_Range_Category Column¶
# In our dataset, we have observed an unusual distribution of values in the Base MSRP column, 
# with a large number of vehicles having a value of 0. This could potentially indicate missing or 
# unknown values in the dataset. To account for this uncertainty and still make use of the available data, 
# we have decided to create a new column called Price_Range_Category based on the Base MSRP values.

# We have defined four categories for the Price_Range_Category column:

# "Unknown": If the Base MSRP value is 0, we assign this category as it might indicate missing or unknown values.
# "Low": If the Base MSRP value is less than 40,000, we assign this category.
# "Medium": If the Base MSRP value is between 40,000 and 60,000, we assign this category.
# "High": If the Base MSRP value is greater than 60,000, we assign this category.
# By creating this new column, we can better understand the distribution of electric vehicle prices 
# in our dataset and account for the potential uncertainty introduced by the large number of 0 values 
# in the Base MSRP column.


# In[20]:


df['Base MSRP'].value_counts()


# In[21]:


def create_price_range_category(df, column='Base MSRP'):
    def categorize_price(price):
        if price == 0:
            return "Unknown"
        elif price < 40000:
            return "Low"
        elif price < 60000:
            return "Medium"
        else:
            return "High"

    df['Price_Range_Category'] = df[column].apply(categorize_price)
    return df

df = create_price_range_category(df, column='Base MSRP')


# In[22]:


# Feature Engineering: Creating an 'Electric_Range_Category' Column¶
# In our dataset, we have observed an unusual distribution of values in the 'Electric Range' column, 
# with a large number of vehicles having a value of 0. This could potentially indicate missing or 
# unknown values in the dataset. To account for this uncertainty and still make use of the 
# available data, we have decided to create a new column called 'Electric_Range_Category' 
# based on the 'Electric Range' values.

# We have defined four categories for the 'Electric_Range_Category' column:

# "Unknown": If the 'Electric Range' value is 0, we assign this category as it might indicate missing 
# or unknown values.

# "Short": If the 'Electric Range' value is less than 150, we assign this category.
# "Medium": If the 'Electric Range' value is between 150 and 300, we assign this category.
# "Long": If the 'Electric Range' value is greater than 300, we assign this category.
# By creating this new column, we can better understand the distribution of electric vehicle 
# ranges in our dataset and account for the potential uncertainty introduced by the large number of 0 values 
# in the 'Electric Range' column.


# In[23]:


df['Electric Range'].value_counts()


# In[24]:


def create_electric_range_category(df, column='Electric Range'):
    def categorize_range(electric_range):
        if electric_range == 0:
            return "Unknown"
        elif electric_range < 150:
            return "Short"
        elif electric_range < 300:
            return "Medium"
        else:
            return "Long"

    df['Electric_Range_Category'] = df[column].apply(categorize_range)
    return df

df = create_electric_range_category(df, column='Electric Range')


# In[25]:


# to view the cleaned and featured-engineered dataframe.


# In[26]:


df.head(5)


# In[27]:


# DATA VISUALIZATION.
# I begin the exloration by visulaizing major viariables in the dataset to see the trends and develop insights 
# in the data.  
# visualization of Electric Range From  1997-2024


# In[28]:


sns.displot(x='Electric Range',data=df)


# In[29]:


#Visualization of Base MSRP

plt.figure(figsize=(10,5))
sns.barplot(y='Base MSRP',x='Make',data=df)

plt.title("Base MSRP vs Make")

plt.xticks(rotation=90)


# In[30]:


sns.barplot(y='Base MSRP',x='Make',data=df)
plt.show()


# In[31]:


# Top most  car models of  electric vehicle

top_most=df['Model'].value_counts().sort_values(ascending=False,).reset_index().head(10)

sns.barplot(y='index',x='Model',data=top_most)

plt.xlabel('Model')

plt.ylabel('No_of_models')

plt.title('Top most  car models of  electric vehicle from 1997-2024. ')


# In[32]:


plt.show()


# In[33]:


# showing visualization of company located in legislative district of state washington

plt.figure(figsize=(10,5))
sns.boxplot(y="Legislative District",x="Make",data=df)
plt.xticks(rotation=90)
plt.show()


# In[34]:


# let see the manufacturing companies thhat makes most of the car model
Companies = df.groupby('Make').count().sort_values(by='City',ascending=False)['City'].index
values = df.groupby('Make').count().sort_values(by='City',ascending=False)['City'].values


# In[35]:


px.pie(names=list(Companies)[:10],values=values[:10],width=500,height=400)


# In[36]:


#Percentage of BEV vs PHEV

Vehicle_type = list(df.groupby('Electric Vehicle Type').count()['County'].index)
values = df.groupby('Electric Vehicle Type').count()['County'].values

px.pie(names=Vehicle_type,values=values,height=400)


# In[37]:



top_10_companies = df['Make'].value_counts().head(10).index
top_10_companies


# In[38]:


#lets see the percentage of top 10 companies vehicles which are BEV and PHEV
for index,i in enumerate(top_10_companies):
    data = df[df['Make']==i]
    labels = list(data.groupby('Electric Vehicle Type').count()['City'].index)
    values = list(data.groupby('Electric Vehicle Type').count()['City'].values)
    fig = px.pie(names=labels,values=values,width=700,height=400,title=str(i))
    fig.show()


# In[39]:


# Top most  car models according to location

top_most=df['Location'].value_counts().sort_values(ascending=False,).reset_index().head(10)

sns.barplot(y='Location',x='index',data=top_most)

plt.xlabel('Location')

plt.ylabel('No_of_models')

plt.xticks(rotation= 90)

plt.title('Top most  car models with respect to location. ')


# In[40]:


plt.show()


# In[41]:


top_most2=df['Make'].value_counts().sort_values(ascending=False,).reset_index().head(10)
top_most2


# In[42]:


# showing the relationship between the longitude and latiude
sns.scatterplot(y='Longitude',x='latitude', data = df)


# In[43]:


plt.show()


# In[44]:


# pairplot

sns.pairplot(data=df[['Postal Code','Model Year','Electric Range','Base MSRP','Legislative District',
                      'DOL Vehicle ID','2020 Census Tract','Longitude','latitude']])


# In[45]:


plt.show()


# In[46]:


# correlation plot.
# In order to visualize the relationship that exist within the numerical # Select only numeric columns (optional)
numeric_df = df[['Postal Code','Model Year','Electric Range','Base MSRP','Legislative District',
                      'DOL Vehicle ID']]

# Calculate correlation matrix
correlation_matrix = numeric_df.corr()

# Create the heatmap
plt.figure(figsize=(11, 8))  # Adjust figure size as needed

# Customize the heatmap using various parameters from Seaborn
sns.heatmap(
    correlation_matrix,
    annot=True,  # Display values within cells
    cmap='coolwarm',  # Color scheme
    fmt=".2f",  # Format values to two decimal places
    linewidths=0.5,  # Line width around cells
    vmin=-1,  # Minimum value for colormap
    vmax=1  # Maximum value for colormap
)

# Add labels and title
plt.title('Correlation Heatmap of Numeric Columns')
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.yticks(rotation=0)  # Keep y-axis labels horizontal

plt.show()
plt.savefig('correlation.png')


# In[47]:


plt.savefig('correlation.png')


# In[48]:


# Data Preparation and processing for machine learning algorithm
# In order for me to facilitate my prediction, I will convert some categorical variables(CAFV and 
# Electric vehicle type) to numerical variable using the label encoder from scikit-learn 


# In[49]:


# FOR A REALISTIC MODEL AND BETTER PERFORMANCE AND ALGORITHM
# since I have too many 0 values in the Base MSRP which is unrealistic value to cost an electric vehicle 
# or any other type of vehicle in the market, 
# I will remove all rows having 0 Base MSRP.


# In[50]:


df_filtered = df[df['Base MSRP'] != 0].copy()


# In[51]:


df_filtered.head(5)


# In[52]:


df_filtered.shape


# In[53]:


# So I will proceed with my machine learning with thhis relaistic data values.


# In[54]:


import sklearn
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[55]:


le = preprocessing.LabelEncoder()
le.fit(df_filtered['CAFV'])
df_filtered.loc[:, 'CAFV'] = le.transform(df_filtered['CAFV'])


# In[56]:


le = preprocessing.LabelEncoder()
le.fit(df_filtered['Electric Vehicle Type'])
df_filtered.loc[:, 'Electric Vehicle Type'] = le.transform(df_filtered['Electric Vehicle Type'])


# In[57]:


le = preprocessing.LabelEncoder()
le.fit(df_filtered['Electric Utility'])
df_filtered.loc[:, 'Electric Utility'] = le.transform(df_filtered['Electric Utility'])


# In[58]:


# Training a linear regression model with the BaseMSRP as the response variable


# In[59]:


lm = LinearRegression()
X = df_filtered[['Postal Code','Model Year','Electric Range','Legislative District','Electric Vehicle Type',
        'CAFV','DOL Vehicle ID', 'Electric Utility']]
y = df_filtered['Base MSRP']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# In[60]:


X


# In[61]:


y


# In[62]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.fit_transform(X_test)


# In[63]:


from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


# In[64]:


def modelresults(predictions):
    print("Mean Absolute error on model is {}".format(mean_absolute_error(y_test, predictions)))
    print("Mean squared error on model is {}".format(np.sqrt(mean_squared_error(y_test, predictions))))


# In[65]:


# get predictions
from sklearn.linear_model import LinearRegression


# In[66]:


lr = LinearRegression()
lr.fit(scaled_X_train, y_train)


# In[67]:


ln_pred = lr.predict(scaled_X_test)


# In[68]:


modelresults(ln_pred)


# In[69]:


# Hyper-parameter tuning with GridSearchCv


# In[70]:


# get predictions
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


# In[71]:


forest = RandomForestRegressor()
param_rf = {
    "max_depth": [3, 10],
    "n_estimators": [2, 4, 6]
}


# In[72]:


grid_search = GridSearchCV(forest, param_rf)


# In[76]:


grid_search.fit(scaled_X_train, y_train)


# In[77]:


grid_search.best_params_


# In[79]:


pred_rf = grid_search.predict(scaled_X_test)


# In[80]:


modelresults(pred_rf)


# In[81]:


# for hyper parameter tuning(to optimize model performance), we use support vector machine with GridSearchCV


# In[82]:


from sklearn.svm import SVR


# In[83]:


param_grdsvr = {"C" : [0.01, 0.1, 0.5], "kernel": ["linear", "rbf", "poly"], "degree": [2, 3, 4]}


# In[84]:


svrmodel = SVR()


# In[85]:


grdsvr = GridSearchCV(svrmodel, param_grdsvr)


# In[86]:


grdsvr.fit(scaled_X_train, y_train)


# In[87]:


grdsvr.best_params_


# In[88]:


pred_grdsvr = grdsvr.predict(scaled_X_test)


# In[89]:


modelresults(pred_grdsvr)


# In[ ]:




