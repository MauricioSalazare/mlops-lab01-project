# Task 1
# Import necessary libraries
import matplotlib
matplotlib.use('Qt5Agg')   # or 'Qt5Agg', 'Agg', 'MacOSX', etc.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Load the Titanic dataset
titanic_data = pd.read_csv('/Users/mauricio.salazar-duque/mlops-lab01-project/MLOps/Data-Files/titanic.csv')
# Display the first few rows of the dataset
print(titanic_data.head())


#%%
# Task 2
# Summary of the dataset
print("\nDataset Information:")
print(titanic_data.info())

# Check for missing values
print("\nMissing values in each column:")
print(titanic_data.isnull().sum())

# Statistical summary
print("\nStatistical Summary:")
print(titanic_data.describe())

#%%
# Check unique values for categorical columns
print(titanic_data['Sex'].unique())
print(titanic_data['Embarked'].unique())
print(titanic_data['Pclass'].unique())


#%%
# Plot histogram for numerical columns
titanic_data.hist(bins=20, figsize=(10, 8))
plt.tight_layout()
plt.show()


#%%
# Scatter plot to visualize Age vs. Fare
plt.figure(figsize=(8,6))
plt.scatter(titanic_data['Age'], titanic_data['Fare'], alpha=0.5)
plt.title('Age vs. Fare')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.show()

#%%
# Boxplot to compare fare by class
plt.figure(figsize=(8,6))
sns.boxplot(x='Pclass', y='Fare', data=titanic_data)
plt.title('Fare Distribution by Passenger Class')
plt.show()


#%%
# Pair plot for continuous variables
sns.pairplot(titanic_data[['Age', 'Fare', 'Pclass']], hue='Pclass', diag_kind='kde')
plt.show()

#%%
# Task 3
# Creating a new feature 'FamilySize'
titanic_data['FamilySize'] = titanic_data['SibSp'] + titanic_data['Parch'] + 1
# Check the first few rows to confirm the new feature
print(titanic_data[['SibSp', 'Parch', 'FamilySize']].head())

#%%
# Extracting title from names
titanic_data['Title'] = titanic_data['Name'].str.extract(' ([A-Za-z]+)', expand=False)

# Check for unique titles
print(titanic_data['Title'].unique())


#%%
# Replace missing values in 'Age' with the median age
titanic_data['Age'] = titanic_data['Age'].fillna(titanic_data['Age'].median())

# Fill missing values in 'Embarked' with the most common port
titanic_data['Embarked'] = titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0])

# Drop the 'Cabin' column since it has too many missing values
titanic_data.drop(columns=['Cabin'], inplace=True)

# Confirm that missing values have been addressed
print("\nMissing values after cleaning:")
print(titanic_data.isnull().sum())


#%%
# Remove outliers in 'Fare'
titanic_data = titanic_data[titanic_data['Fare'] < 300]

# Plot the updated boxplot for Fare by Passenger Class
plt.figure(figsize=(8,6))
sns.boxplot(x='Pclass', y='Fare', data=titanic_data)
plt.title('Fare Distribution by Passenger Class (Outliers Removed)')
plt.show()

#%%
# Save the cleaned dataset to a new CSV file
titanic_data.to_csv('data_processed/titanic_cleaned.csv', index=False)
print("Cleaned dataset saved as 'titanic_cleaned.csv'")







#%%
titanic_cleaned = pd.read_csv('data_processed/titanic_cleaned.csv')
titanic_cleaned.head()


#%%
# Countplot for Pclass
sns.countplot(x='Pclass', data=titanic_cleaned)
plt.title('Count of Passengers by Class')
plt.show()


# Countplot for Survived
sns.countplot(x='Survived', data=titanic_data)
plt.title('Count of Survived vs Not Survived')
plt.show()


#%%
# Violin plot for Age distribution by survival status
sns.violinplot(x='Survived', y='Age', data=titanic_cleaned)
plt.title('Age Distribution by Survival Status')
plt.show()
