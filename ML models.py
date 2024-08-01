#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Decision Tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Test the classifier
y_pred = clf.predict(X_test)
print(f"Model accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# Function to input new flower data and predict species
def predict_species(new_flower_data):
    prediction = clf.predict([new_flower_data])
    species = iris.target_names[prediction][0]
    return species

# Input new flower data
new_flower_data = [5.1, 3.5, 1.4, 0.2]  # Example input: sepal length, sepal width, petal length, petal width

# Predict the species
species_prediction = predict_species(new_flower_data)
print(f"The predicted species of the iris flower is: {species_prediction}")


# In[2]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Define the dataset
data = {
    'fever': [101, 98, 100, 99, 102],
    'cough': [1, 0, 1, 0, 1],
    'shortness_of_breath': [0, 0, 1, 1, 0],
    'age': [45, 34, 65, 50, 70],
    'condition': [1, 0, 1, 0, 1]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Features and target variable
X = df.drop('condition', axis=1)
y = df['condition']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Function to input new patient data and predict condition
def predict_condition(new_patient_data, k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    
    # Standardize the new patient data
    new_patient_data = scaler.transform([new_patient_data])
    
    prediction = knn.predict(new_patient_data)
    return prediction[0]

# Input new patient data
new_patient_data = [99, 1, 0, 55]  # Example input features: fever, cough, shortness_of_breath, age
k = 3  # Number of neighbors

# Predict the condition
condition_prediction = predict_condition(new_patient_data, k)

# Output the result
if condition_prediction == 1:
    print("The patient is predicted to have the medical condition.")
else:
    print("The patient is predicted not to have the medical condition.")


# In[4]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Define the dataset
data = {
    'area': [1500, 2000, 2500, 1800, 3000, 2400, 2200, 3500, 1600, 2800],
    'bedrooms': [3, 4, 3, 3, 5, 4, 4, 5, 2, 4],
    'location': [1, 2, 3, 1, 2, 3, 1, 2, 1, 3],  # Encoded locations
    'price': [300000, 400000, 500000, 350000, 600000, 450000, 420000, 550000, 320000, 480000]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Features and target variable
X = df.drop('price', axis=1)
y = df['price']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the Linear Regression model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Function to input new house data and predict price
def predict_price(new_house_data):
    # Standardize the new house data
    new_house_data = scaler.transform([new_house_data])
    
    prediction = lr.predict(new_house_data)
    return prediction[0]

# Input new house data
new_house_data = [2000, 3, 2]  # Example input features: area, bedrooms, location

# Predict the price
price_prediction = predict_price(new_house_data)
print(f"The predicted price of the house is: ${price_prediction:.2f}")


# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Define the dataset
data = {
    'usage_minutes': [300, 400, 200, 450, 500, 350, 600, 700, 250, 480],
    'contract_duration': [12, 24, 6, 18, 24, 12, 24, 36, 6, 18],
    'customer_service_calls': [1, 2, 1, 3, 1, 1, 2, 3, 2, 2],
    'churn': [0, 1, 0, 1, 1, 0, 1, 1, 0, 1]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Features and target variable
X = df.drop('churn', axis=1)
y = df['churn']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the Logistic Regression model
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Function to input new customer data and predict churn
def predict_churn(new_customer_data):
    # Standardize the new customer data
    new_customer_data = scaler.transform([new_customer_data])
    
    prediction = lr.predict(new_customer_data)
    return prediction[0]

# Input new customer data
new_customer_data = [350, 12, 1]  # Example input features: usage minutes, contract duration, customer service calls

# Predict the churn
churn_prediction = predict_churn(new_customer_data)
if churn_prediction == 1:
    print("The customer is predicted to churn.")
else:
    print("The customer is predicted not to churn.")

