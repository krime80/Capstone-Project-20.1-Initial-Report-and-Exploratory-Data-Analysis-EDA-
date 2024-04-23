#!/usr/bin/env python
# coding: utf-8

# ## Data Exploration

# In[22]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define File Path
file_path = "/Users/krista.rime/Documents/AIML/capstone_project.csv"

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)


# In[23]:


# Display the first few rows of the DataFrame
print("First few rows of the DataFrame:")
print(df.head())


# In[24]:


# Display summary statistics of numerical features
print("\nSummary statistics of numerical features:")
print(df.describe())


# In[25]:


# Check for missing values
print("\nMissing values in the DataFrame:")
print(df.isnull().sum())


# In[26]:


# Plot histograms of numerical features
print("\nHistograms of numerical features:")
df.hist(figsize=(10, 8))
plt.tight_layout()
plt.show()


# In[32]:


# Display counts of unique values for numerical features in a table
print("\nCounts of unique values for numerical features:")
for feature in numeric_features:
    unique_counts = df[feature].value_counts().reset_index().rename(columns={feature: 'Count', 'index': feature})
    print(unique_counts)


# In[33]:


# Plot counts of categorical features
print("\nCounts of categorical features:")
categorical_features = df.select_dtypes(include=['object']).columns
for feature in categorical_features:
    plt.figure(figsize=(8, 6))
    sns.countplot(x=feature, data=df, order=df[feature].value_counts().index)
    plt.title(f'Count of {feature}')
    plt.xticks(rotation=45)
    plt.show()


# In[34]:


# Display counts of unique values for categorical features in a table
print("\nCounts of unique values for categorical features:")
for feature in categorical_features:
    unique_counts = df[feature].value_counts().reset_index().rename(columns={feature: 'Count', 'index': feature})
    print(unique_counts)


# In[41]:


# Explore relationships between variables using scatter plots or pair plots
print("\nRelationships between variables:")
sns.pairplot(df, diag_kind='kde')
plt.tight_layout()
plt.show()


# In[46]:


# Explore relationships between variables using correlation table
print("\nCorrelation table between numerical features:")
numerical_features = df.select_dtypes(include=['float64', 'int64']).columns
corr_table = df[numerical_features].corr()
print(corr_table)


# In[29]:


# Plot a heatmap of correlations between numerical features
print("\nHeatmap of correlations between numerical features:")
numeric_df = df.select_dtypes(include=['float64', 'int64'])
corr_matrix = numeric_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()


# In[37]:


# Explore relationships between numerical variables using correlation table
print("\nCorrelation table between numerical features:")
numerical_df = df.select_dtypes(include=['float64', 'int64'])
corr_table = numerical_df.corr()
print(corr_table)


# In[ ]:




