#!/usr/bin/env python
# coding: utf-8

# # {Project Title}📝
# 
# ![Banner](./assets/banner.jpeg)

# ## Topic
# *What problem are you (or your stakeholder) trying to address?*
# 📝 <!-- Answer Below -->
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

# In[ ]:


https://www.kaggle.com/datasets/pralabhpoudel/world-energy-consumption
https://www.kaggle.com/datasets/anshtanwar/global-data-on-sustainable-energy
#I have found only 2 datasets that would make sense to me to use at the moment. I have found a dataset on World Energy Consumption and Data on sustainable energy.


# In[ ]:





# ## Approach and Analysis
# *What is your approach to answering your project question?*
# *How will you use the identified data to answer your project question?*
# 📝 <!-- Start Discussing the project here; you can add as many code cells as you need -->

# In[1]:


#Energy Consumption and Efficiency Analysis: Assess the energy consumption of buildings, industries, or households over time.
#Calculate energy efficiency and identify opportunities for energy conservation.
#Analyze the impact of energy-efficient technologies and practices.

#Renewable Energy Adoption:Investigate the growth of renewable energy sources like solar, wind, and hydroelectric power.
#Analyze factors influencing the adoption of renewable energy technologies.
#Assess the environmental and economic benefits of renewable energy.

#Carbon Emissions and Climate Change: Study historical and current carbon emissions data.
#Assess the impact of emissions on global and regional climate change.
#Analyze the effectiveness of carbon reduction policies and practices.


# ## Resources and References
# *What resources and references have you used for this project?*
# 📝 <!-- Answer Below -->

# In[1]:


# ⚠️ Make sure you run this cell at the end of your notebook before every submission!
get_ipython().system('jupyter nbconvert --to python source.ipynb')


# In[18]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[20]:


data = pd.read_csv('data/global-data-on-sustainable-energy (1).csv')


# In[21]:


print(data.head())  # View the first few rows of the dataset
print(data.info())  # Get information about columns and data types
print(data.describe())  # Summary statistics for numerical columns


# In[22]:


# Check for missing values
print(data.isnull().sum())



# In[23]:


# Check for duplicate rows
print(data.duplicated().sum())


# In[24]:


data_without_nan = data.dropna()
print(data_without_nan.isnull().sum())


# In[25]:


print(data.columns)


# In[26]:


selected_columns = ['Electricity from fossil fuels (TWh)', 'Electricity from nuclear (TWh)', 'Electricity from renewables (TWh)']  # Replace with the columns you want

# Extract the selected columns from the DataFrame
selected_data = data[selected_columns]

# Calculate the correlation matrix for the selected columns
corr_matrix = selected_data.corr()

# Display the correlation matrix
print(corr_matrix)


# In[27]:


plt.figure(figsize=(8, 6))  # Set the size of the figure
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix Heatmap')
plt.show()


# In[28]:


# There's a notable positive relationship between electricity generation from fossil fuels and renewables, suggesting some degree of dependency or association between them. Nuclear energy generation shows a moderate positive relationship with electricity generation from fossil fuels and a weaker relationship with renewables.


# In[29]:


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


# In[30]:


# This was intersting to see that fossil fuels, something we will eventually run out of, has been increasing tremendously since 2000. Renewable energy is not something that seems to put into consideration.


# In[31]:


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


# In[32]:


# When we pair this with the energy consumption trend over time, we know that fossil fuel is a favorite when it comes to generating electricity, but that also means the CO2 emmisions from burning fossil fuels is not good for the environment at all.


# In[58]:


grouped_by_country = data.groupby('Entity')['Value_co2_emissions_kt_by_country'].sum().reset_index()

# Selecting the top 3 countries with the highest total CO2 emissions
top_3_emitting_countries = grouped_by_country.nlargest(3, 'Value_co2_emissions_kt_by_country')

print(top_3_emitting_countries)


# In[ ]:


# The value of co2 emissions is in scientific notation. For example if we look at China, 1.527328e+08 KT, that is equivalent to 152,732,800 kilotons of CO2 emissions. 
# My next graph will be CO2 emissions for the top 3 countries with the highest emissions. This visualization could help observe the trajectory of CO2 emissions over time.


# In[60]:


countries_to_plot = ['China', 'United States', 'India']  # Replace with your desired country names

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


# In[61]:


# Based on these trends, it seems like the United States is doing much better than India and China in terms of controlling CO2 emissions. It is still extremely high, but it has a down trend rather then an up trend like China and India


# In[64]:


# So far, what I have completed is, 
# Calculated statistical summaries for numerical columns like CO2 emissions using Pandas' describe() method.
# Obtained value counts or frequency distribution for categorical columns using Pandas' value_counts() method.
# Created histograms or boxplots to visualize distributions of numerical columns like CO2 emissions, energy consumption, etc., using Matplotlib or Seaborn.
# Generated bar charts or count plots for categorical data to display the distribution of categories using Matplotlib or Seaborn.
# Calculated and visualized a correlation matrix to understand relationships between numerical features using Pandas and Seaborn.
# dentified and handled missing values using Pandas' isnull() or dropna() method to decide whether to impute or remove them.
# Checked and transformed data types of columns using Pandas' dtypes attribute and astype() method for conversions if needed.


# In[65]:


# These examples cover various aspects of EDA by performing operations such as summary statistics, distribution visualizations, correlation analysis, handling missing values, outliers, and data type transformations.


# In[66]:


# Types of machine learning I can possibly use.
# Regression Models: Predicting quantitative values such as energy consumption, CO2 emissions, or renewable energy production using linear regression, polynomial regression, or other regression techniques.
# Classification Models: Categorizing or classifying energy-related scenarios, like classifying countries based on energy sources or predicting whether a country has access to clean fuels for cooking.


# In[ ]:


# Issues and Challenges
# Issues related to missing data, outliers, or inconsistencies might impact model performance. 
# Different scales or distributions of features might affect the performance of some algorithms.
# Guarding against overfitting by using proper cross-validation techniques and selecting appropriate evaluation metrics to measure model performance accurately.

