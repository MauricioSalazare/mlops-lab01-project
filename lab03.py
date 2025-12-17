# Import necessary libraries
import matplotlib
matplotlib.use('Qt5Agg')   # or 'Qt5Agg', 'Agg', 'MacOSX', etc.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib


# Load the cleaned Titanic dataset
data = pd.read_csv('data_processed/titanic_cleaned.csv')

# Display the first few rows of the dataset
print("First 5 rows of the dataset:")
print(data.head())

#%%
# Separate features and target variable
X = data.drop(['Survived'], axis=1)
y = data['Survived']


#%%
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



#%%
# Task 2
# Select categorical and numerical columns for preprocessing
categorical_cols = ['Pclass', 'Sex', 'Embarked']
numerical_cols = ['Age', 'Fare', 'FamilySize']

#%%

# Define preprocessing for numeric features (scaling) and categorical features (one-hot encoding)
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Create a column transformer for applying transformations
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])


#%%
# Feature selection: Select top k features
feature_selector = SelectKBest(score_func=f_classif, k=8)


#%%
# Define the model
model = RandomForestClassifier(random_state=42)


#%%

# Create a pipeline that combines the preprocessor, feature selector, and the model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('feature_selector', feature_selector),
    ('classifier', model)
])


#%%
# Train the model
pipeline.fit(X_train, y_train)



#%%
# Predict on the test set
y_pred = pipeline.predict(X_test)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

#%%
# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


#%%

# Save the trained pipeline to a file
joblib.dump(pipeline, 'basic_titanic_pipeline.pkl')
print("Trained model saved as 'basic_titanic_pipeline.pkl'")


#%%
print(pipeline.named_steps['preprocessor'].get_feature_names_out())
