#!/usr/bin/env python
# coding: utf-8

# # {Project Title}📝
# 
# ![Banner](./assets/banner.jpeg)

# ## Topic
# 
# The problem we are addressing is total energy consumption vs energy output

# ## Project Question
# *What specific question are you seeking to answer with this project?*
# *This is not the same as the questions you ask to limit the scope of the project.*
# 📝 <!-- Answer Below -->
# On our trajectory (the world) how long can we really last with how we manage our energy. Is it really sustainable? Are there better decisions to be made regarding how to acquire green energy and reuse it.

# ## What would an answer look like?
# *What is your hypothesized answer to your question?*
# 📝 There are many more sustainable ways to harvest energy. We will eliminate fossil fuels and operate on full green energy such as wind and solar.

# ## Data Sources
# *What 3 data sources have you identified for this project?*
# *How are you going to relate these datasets?*
# 📝 <!-- Answer Below -->

# https://www.kaggle.com/datasets/pralabhpoudel/world-energy-consumption
# https://www.kaggle.com/datasets/anshtanwar/global-data-on-sustainable-energy
# https://www.kaggle.com/datasets/soheiltehranipour/co2-dataset-in-usa 
# I have found only 2 datasets that would make sense to me to use at the moment. I have found a dataset on World Energy Consumption and Data on sustainable energy.

# ## Approach and Analysis
# *What is your approach to answering your project question?*
# *How will you use the identified data to answer your project question?*
# 📝 <!-- Start Discussing the project here; you can add as many code cells as you need -->

# Energy Consumption and Efficiency Analysis: Assess the energy consumption of buildings, industries, or households over time.
# Calculate energy efficiency and identify opportunities for energy conservation.
# Analyze the impact of energy-efficient technologies and practices.
# Renewable Energy Adoption:Investigate the growth of renewable energy sources like solar, wind, and hydroelectric power.
# Analyze factors influencing the adoption of renewable energy technologies.
# Assess the environmental and economic benefits of renewable energy.
# 
# Carbon Emissions and Climate Change: Study historical and current carbon emissions data.
# Assess the impact of emissions on global and regional climate change.
# Analyze the effectiveness of carbon reduction policies and practices.

# ## Resources and References
# *What resources and references have you used for this project?*
# 📝 <!-- Answer Below -->

# In[50]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


# In[51]:


data = pd.read_csv('data/global-data-on-sustainable-energy (1).csv')


# In[52]:


print(data.head())  # View the first few rows of the dataset
print(data.info())  # Get information about columns and data types
print(data.describe())  # Summary statistics for numerical columns


# In[53]:


# Check for missing values
print(data.isnull().sum())



# In[54]:


# Check for duplicate rows
print(data.duplicated().sum())


# In[55]:


data_without_nan = data.dropna()
print(data_without_nan.isnull().sum())


# In[56]:


print(data.columns)


# In[57]:


selected_columns = ['Electricity from fossil fuels (TWh)', 'Electricity from nuclear (TWh)', 'Electricity from renewables (TWh)']

# Extract the selected columns from the DataFrame
selected_data = data[selected_columns]

# Calculate the correlation matrix for the selected columns
corr_matrix = selected_data.corr()

# Display the correlation matrix
print(corr_matrix)


# # There's a notable positive relationship between electricity generation from fossil fuels and renewables, suggesting some degree of dependency or association between them. Nuclear energy generation shows a moderate positive relationship with electricity generation from fossil fuels and a weaker relationship with renewables.

# In[58]:


plt.figure(figsize=(8, 6))  # Set the size of the figure
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix Heatmap')
plt.show()


# In[60]:


energy_trends = data[['Year', 'Electricity from fossil fuels (TWh)', 'Electricity from nuclear (TWh)', 'Electricity from renewables (TWh)']]

# Set the 'Year' column as the index for easy plotting
energy_trends.set_index('Year', inplace=True)

# Plotting the trends over time for energy sources
plt.figure(figsize=(10, 6))
for column in energy_trends.columns:
    plt.plot(energy_trends.index, energy_trends[column], label=column)

plt.xlabel('Year')
plt.ylabel('Electricity Generation (TWh)')
plt.title('Energy Consumption Trends Over Time')
plt.legend()
plt.grid(True)
plt.show()


#  This was intersting to see that fossil fuels, something we will eventually run out of, has been increasing tremendously since 2000. Renewable energy is not something that seems to put into consideration.

# In[61]:


co2_emissions = data['Value_co2_emissions_kt_by_country']
electricity_fossil_fuels = data['Electricity from fossil fuels (TWh)']

# Plotting CO2 emissions against electricity generation from fossil fuels
plt.figure(figsize=(8, 6))
plt.scatter(electricity_fossil_fuels, co2_emissions, alpha=0.5)
plt.xlabel('Electricity from fossil fuels (TWh)')
plt.ylabel('CO2 Emissions (kt)')
plt.title('CO2 Emissions vs Electricity from Fossil Fuels')
plt.grid(True)
plt.show()


#  When we pair this with the energy consumption trend over time, we know that fossil fuel is a favorite when it comes to generating electricity, but that also means the CO2 emmisions from burning fossil fuels is not good for the environment at all.

# In[62]:


grouped_by_country = data.groupby('Entity')['Value_co2_emissions_kt_by_country'].sum().reset_index()

# Selecting the top 3 countries with the highest total CO2 emissions
top_10_emitting_countries = grouped_by_country.nlargest(10, 'Value_co2_emissions_kt_by_country')

print(top_10_emitting_countries)


# The value of co2 emissions is in scientific notation. For example if we look at China, 1.527328e+08 KT, that is equivalent to 152,732,800 kilotons of CO2 emissions. 
# My next graph will be CO2 emissions for the top 3 countries with the highest emissions. This visualization could help observe the trajectory of CO2 emissions over time.

# In[64]:


countries_to_plot = ['China', 'United States', 'India']

# Filtering data for selected countries
selected_countries_data = data[data['Entity'].isin(countries_to_plot)]

# Plotting CO2 emissions over the years for selected countries
plt.figure(figsize=(10, 6))
for country in countries_to_plot:
    country_data = selected_countries_data[selected_countries_data['Entity'] == country]
    plt.plot(country_data['Year'], country_data['Value_co2_emissions_kt_by_country'], label=country)

plt.xlabel('Year')
plt.ylabel('CO2 Emissions (kt)')
plt.title('CO2 Emissions Over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


#  Based on these trends, it seems like the United States is doing much better than India and China in terms of controlling CO2 emissions. It is still extremely high, but it has a down trend rather then an up trend like China and India

#  So far, what I have completed is, 
#  Calculated statistical summaries for numerical columns like CO2 emissions using Pandas' describe() method.
#  Obtained value counts or frequency distribution for categorical columns using Pandas' value_counts() method.
#  Created histograms or boxplots to visualize distributions of numerical columns like CO2 emissions, energy consumption, etc., using Matplotlib or Seaborn.
#  Generated bar charts or count plots for categorical data to display the distribution of categories using Matplotlib or Seaborn.
#  Calculated and visualized a correlation matrix to understand relationships between numerical features using Pandas and Seaborn.
#  dentified and handled missing values using Pandas' isnull() or dropna() method to decide whether to impute or remove them.
#  Checked and transformed data types of columns using Pandas' dtypes attribute and astype() method for conversions if needed.
# 

#  These examples cover various aspects of EDA by performing operations such as summary statistics, distribution visualizations, correlation analysis, handling missing values, outliers, and data type transformations.

#  Types of machine learning I can possibly use.
#  Regression Models: Predicting quantitative values such as energy consumption, CO2 emissions, or renewable energy production using linear regression, polynomial regression, or other regression techniques.
# 

# Issues and Challenges
# Issues related to missing data, outliers, or inconsistencies might impact model performance. 
# Different scales or distributions of features might affect the performance of some algorithms.
# Guarding against overfitting by using proper cross-validation techniques and selecting appropriate evaluation metrics to measure model performance accurately.
# 

# We begin Machine learning
# Will start by preprocessing the data, like checking for missing values.

# In[65]:


c02 = pd.read_csv('data/co2.csv')
c02.head(5)


# In[66]:


c02.info()


# In[67]:


c02['Month'] = c02.YYYYMM.astype(str).str[4:6].astype(float)
c02['Year'] = c02.YYYYMM.astype(str).str[0:4].astype(float)


# In[68]:


c02.shape


# In[69]:


c02.drop(['YYYYMM'], axis=1, inplace=True)
c02.replace([np.inf, -np.inf], np.nan, inplace=True)
c02.tail(5)


# We use Pandas to import the CSV file. We notice that the dataframe contains a column 'YYYYMM' that needs to be separated into 'Year' and 'Month' column. In this step, we will also remove any null values that we may have in the dataframe. Finally, we will retrieve the last five elements of the dataframe to check if our code worked. And it did!

# In[70]:


print(c02.dtypes)


# In[71]:


c02.isnull().sum()


# In[72]:


c02.shape


# In[73]:


X = c02.loc[:,['Month', 'Year']].values
y = c02.loc[:,'Value'].values


# In[74]:


y


# In[75]:


c02_dmatrix = xgb.DMatrix(X,label=y)


# In[76]:


c02_dmatrix


# In[77]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[78]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[79]:


reg_mod = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.08,
    subsample=0.75,
    colsample_bytree=1, 
    max_depth=7,
    gamma=0,
)
reg_mod.fit(X_train, y_train)


# After training the model, we'll check the model training score.

# In[80]:


scores = cross_val_score(reg_mod, X_train, y_train,cv=10)
print("Mean cross-validation score: %.2f" % scores.mean())


# the model predicted the target variable very accurately across the different subsets of the data used for training and testing. This high score generally indicates that the model is performing well and is able to generalize effectively to unseen data.

# In[81]:


reg_mod.fit(X_train,y_train)

predictions = reg_mod.predict(X_test)


# In[83]:


rmse = np.sqrt(mean_squared_error(y_test, predictions))
print("RMSE: %f" % (rmse))


# In[85]:


from sklearn.metrics import r2_score
r2 = np.sqrt(r2_score(y_test, predictions))
print("R_Squared Score : %f" % (r2))


# As you can see, the these statistical metrics have reinstated our confidence about this model. RMSE ~ 4.95 R-Squared Score ~ 98.8% Now, let's visualize the original data set using the seaborn library.

# In[86]:


plt.figure(figsize=(10, 5), dpi=80)
sns.lineplot(x='Year', y='Value', data=c02)


# In[87]:


plt.figure(figsize=(10, 5), dpi=80)
x_ax = range(len(y_test))
plt.plot(x_ax, y_test, label="test")
plt.plot(x_ax, predictions, label="predicted")
plt.title("Carbon Dioxide Emissions - Test and Predicted data")
plt.legend()
plt.show()


# Finally, the last piece of code will print the forecasted carbon dioxide emissions until 2025.

# In[88]:


plt.figure(figsize=(10, 5), dpi=80)
df=pd.DataFrame(predictions, columns=['pred']) 
df['date'] = pd.date_range(start='8/1/2016', periods=len(df), freq='M')
sns.lineplot(x='date', y='pred', data=df)
plt.title("Carbon Dioxide Emissions - Forecast")
plt.show()

