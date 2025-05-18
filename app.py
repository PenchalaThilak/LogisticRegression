# Import necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
column_names = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
]

data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
                   header=None, names=column_names, na_values=" ?", sep=",\s*", engine='python')

# Display the first few rows
data.head()

# Shape of the dataset
data.shape

# Check for missing values
data.isnull().sum()

# Drop rows with missing values
data.dropna(inplace=True)

# Shape after dropping NA
data.shape

# Visualizing count of people by income
sns.countplot(x='income', data=data)
plt.title("Income Distribution")
plt.show()

# Age distribution
sns.histplot(data['age'], bins=20, kde=True)
plt.title("Age Distribution")
plt.show()

# Education count
sns.countplot(y='education', data=data, order=data['education'].value_counts().index)
plt.title("Education Count")
plt.show()

# Encode categorical variables
data_encoded = pd.get_dummies(data, drop_first=True)

# Import train_test_split and logistic regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Define features and target
X = data_encoded.drop('income_>50K', axis=1)
y = data_encoded['income_>50K']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Accuracy score
print("Accuracy:", accuracy_score(y_test, y_pred))

# Classification report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
conf_mat = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
