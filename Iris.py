import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt 

#Load the Iris dataset from seaborn
iris = sns.load_dataset('iris')

#Display the first few rows of the dataset
print(iris.head())

#Display the summary of the dataset
print(iris.info())

#Check for missing values
print(iris.isnull().sum())

# Display summary statistics
print(iris.describe())

#pairplot to show scatterplots for each pair of features and histograms for each feature
sns.pairplot(iris, hue='species')
plt.show()

#heatmap to visualize the correlation between different features
#Compute the correlation matrix
corr_matrix = iris.corr()
print(corr_matrix)

#Generate a heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

#Boxplots for each feature grouped by species will help us understand the distribution and outliers.
fig, axes = plt.subplots(2, 2, figsize=(12,8))
sns.boxplot(x='species', y='sepal_length', data=iris, ax=axes[0,0])
sns.boxplot(x='species', y='sepal_width', data=iris, ax=axes[0,1])
sns.boxplot(x='species', y='petal_length', data=iris, ax=axes[1,0])
sns.boxplot(x='species', y='petal_width', data=iris, ax=axes[1,1])
plt.tight_layout()
plt.show()

# Violin plots will give us a combined view of the KDE and boxplot.
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
sns.violinplot(x='species', y='sepal_length', data=iris, ax=axes[0,0])
sns.violinplot(x='species', y='sepal_width', data=iris, ax=axes[0,1])
sns.violinplot(x='species', y='petal_length', data=iris, ax=axes[1,0])
sns.violinplot(x='species', y='petal_width', data=iris, ax=axes[1,1])
plt.tight_layout()
plt.show()
