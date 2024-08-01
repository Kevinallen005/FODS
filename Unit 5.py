# -*- coding: utf-8 -*-
"""Untitled7.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1G23nxzM9ts-mLtU5WnzaVlpRPHJRSga1
"""

import numpy as np

# Create the student_scores array
student_scores = np.array([
    [85, 90, 78, 92],
    [88, 76, 85, 85],
    [92, 88, 82, 89],
    [75, 85, 80, 87]
])

# Define the subject names
subjects = ['Math', 'Science', 'English', 'History']

# Calculate the average score for each subject
average_scores = student_scores.mean(axis=0)
print("Average Scores for Each Subject:", average_scores)

# Identify the subject with the highest average score
highest_avg_score_index = np.argmax(average_scores)
highest_avg_score_subject = subjects[highest_avg_score_index]
print("Subject with the Highest Average Score:", highest_avg_score_subject)

import numpy as np

# Example 3x3 NumPy array representing sales data for three products
# Each row represents a different product and each element represents a sale price
sales_data = np.array([[100, 200, 300],
                       [150, 250, 350],
                       [180, 280, 380]])

# Calculate the average price of all products sold
average_price = np.mean(sales_data)

print(f"The average price of all products sold in the past month is: ${average_price:.2f}")

import numpy as np

# Create the student_scores array
student_scores = np.array([
    [85, 90, 78, 92],  # Scores for Student 1
    [88, 76, 85, 85],  # Scores for Student 2
    [92, 88, 82, 89],  # Scores for Student 3
    [75, 85, 80, 87]   # Scores for Student 4
])

# Define the subject names
subjects = ['Math', 'Science', 'English', 'History']

# Calculate the average score for each subject
average_scores = student_scores.mean(axis=0)
print("Average Scores for Each Subject:", average_scores)

# Identify the subject with the highest average score
highest_avg_score_index = np.argmax(average_scores)
highest_avg_score_subject = subjects[highest_avg_score_index]
print("Subject with the Highest Average Score:", highest_avg_score_subject)

import numpy as np

# Create the sales_data array
sales_data = np.array([
    [100, 150, 200],  # Sales data for Product 1
    [250, 300, 350],  # Sales data for Product 2
    [400, 450, 500]   # Sales data for Product 3
])

# Calculate the average price of all products sold
average_price = np.mean(sales_data)
print("Average Price of All Products Sold:", average_price)

import numpy as np
import pandas as pd

# Create a sample NumPy array (4 rows and 4 columns)
house_data = np.array([
    [3, 1500, 20, 300000],  # [bedrooms, square_footage, age, sale_price]
    [4, 2000, 15, 350000],
    [2, 1200, 30, 250000],
    [3, 1800, 10, 320000]
])

# Define column names
columns = ['Bedrooms', 'Square_Footage', 'Age', 'Sale_Price']

# Convert NumPy array to DataFrame
df = pd.DataFrame(house_data, columns=columns)

# Display the first few rows of the DataFrame
print("DataFrame Head:")
print(df.head())

# Calculate and print basic statistics for sale prices
mean_price = df['Sale_Price'].mean()
median_price = df['Sale_Price'].median()
std_dev_price = df['Sale_Price'].std()

print("\nBasic Statistics for Sale Prices:")
print("Mean Sale Price:", mean_price)
print("Median Sale Price:", median_price)
print("Standard Deviation of Sale Prices:", std_dev_price)

# Calculate and print average square footage
average_square_footage = df['Square_Footage'].mean()
print("\nAverage Square Footage:", average_square_footage)

# Calculate and print correlation between square footage and sale price
correlation = df['Square_Footage'].corr(df['Sale_Price'])
print("Correlation between Square Footage and Sale Price:", correlation)

import numpy as np

# Create a sample NumPy array with quarterly sales data
sales_data = np.array([15000, 20000, 25000, 30000])

# Calculate the total sales for the year
total_sales = np.sum(sales_data)
print("Total Sales for the Year:", total_sales)

# Calculate the percentage increase from the first quarter to the fourth quarter
sales_q1 = sales_data[0]  # Sales in the first quarter
sales_q4 = sales_data[3]  # Sales in the fourth quarter

percentage_increase = ((sales_q4 - sales_q1) / sales_q1) * 100
print("Percentage Increase from Q1 to Q4:", percentage_increase)

import numpy as np

# Create a sample NumPy array with fuel efficiency data (in miles per gallon)
fuel_efficiency = np.array([25.0, 30.0, 22.5, 27.5, 31.0, 29.5])

# Calculate the average fuel efficiency
average_fuel_efficiency = np.mean(fuel_efficiency)
print("Average Fuel Efficiency:", average_fuel_efficiency)

# Assume you want to compare the fuel efficiency of the first and last car models
fuel_efficiency_model1 = fuel_efficiency[0]  # Fuel efficiency of the first model
fuel_efficiency_model2 = fuel_efficiency[-1] # Fuel efficiency of the last model

# Calculate the percentage improvement
percentage_improvement = ((fuel_efficiency_model2 - fuel_efficiency_model1) / fuel_efficiency_model1) * 100
print("Percentage Improvement in Fuel Efficiency from Model 1 to Model 2:", percentage_improvement)

import pandas as pd

# Load the dataset into a Pandas DataFrame
df = pd.read_csv('employee_data.csv')

# Display the first few rows of the DataFrame
print("DataFrame Head:")
print(df.head())

# Convert the 'JoiningDate' column to datetime
df['JoiningDate'] = pd.to_datetime(df['JoiningDate'])

# 1. Determine the highest and lowest salaries in each department
salary_stats = df.groupby('Department')['Salary'].agg(['max', 'min'])
print("\nHighest and Lowest Salaries in Each Department:")
print(salary_stats)

# 2. Calculate the average tenure of employees in the company
# Current date
current_date = pd.to_datetime('today')

# Calculate tenure in days
df['Tenure'] = (current_date - df['JoiningDate']).dt.days

# Calculate average tenure
average_tenure = df['Tenure'].mean()
print("\nAverage Tenure of Employees in Days:", average_tenure)

# 3. Identify employees who joined before a specific date
specific_date = pd.to_datetime('2022-01-01')
employees_before_date = df[df['JoiningDate'] < specific_date]
print(f"\nEmployees who joined before {specific_date.date()}:")
print(employees_before_date[['EmployeeID', 'Department', 'JoiningDate']])

import pandas as pd

# Sample data for demonstration purposes
data = {
    'EmployeeID': [101, 102, 103, 104, 105],
    'Department': ['HR', 'Finance', 'IT', 'HR', 'IT'],
    'Salary': [50000, 60000, 55000, 52000, 58000],
    'JoiningDate': ['2021-06-01', '2020-03-15', '2022-01-10', '2021-11-25', '2021-05-30']
}

# Create a DataFrame from the sample data
df = pd.DataFrame(data)

# Convert 'JoiningDate' column to datetime
df['JoiningDate'] = pd.to_datetime(df['JoiningDate'])

# 1. Determine the highest and lowest salaries in each department
salary_stats = df.groupby('Department')['Salary'].agg(['max', 'min'])
print("\nHighest and Lowest Salaries in Each Department:")
print(salary_stats)

# 2. Calculate the average tenure of employees in the company
# Current date
current_date = pd.to_datetime('today')

# Calculate tenure in days
df['Tenure'] = (current_date - df['JoiningDate']).dt.days

# Calculate average tenure
average_tenure = df['Tenure'].mean()
print("\nAverage Tenure of Employees in Days:", average_tenure)

# 3. Identify employees who joined before a specific date
specific_date = pd.to_datetime('2022-01-01')
employees_before_date = df[df['JoiningDate'] < specific_date]
print(f"\nEmployees who joined before {specific_date.date()}:")
print(employees_before_date[['EmployeeID', 'Department', 'JoiningDate']])

import pandas as pd

# Sample data for recovery times in days
recovery_times = pd.Series([5, 7, 8, 6, 9, 10, 12, 14, 16, 20])

# Calculate the 10th, 50th, and 90th percentiles
percentiles = recovery_times.quantile([0.10, 0.50, 0.90])
print("Percentiles using Pandas:")
print(percentiles)
import numpy as np

# Sample data for recovery times in days
recovery_times = np.array([5, 7, 8, 6, 9, 10, 12, 14, 16, 20])

# Calculate the 10th, 50th, and 90th percentiles
percentiles = np.percentile(recovery_times, [10, 50, 90])
print("Percentiles using NumPy:")
print("10th percentile:", percentiles[0])
print("50th percentile (Median):", percentiles[1])
print("90th percentile:", percentiles[2])

import pandas as pd

# Sample data for purchase amounts in dollars
purchase_amounts = pd.Series([25.0, 30.0, 25.0, 40.0, 30.0, 50.0, 60.0, 30.0, 20.0, 25.0])

# Calculate the mean (average) purchase amount
mean_purchase_amount = purchase_amounts.mean()
print("Mean Purchase Amount:", mean_purchase_amount)

# Identify the mode of the purchase amounts
mode_purchase_amount = purchase_amounts.mode()
print("Mode of Purchase Amounts:", mode_purchase_amount.iloc[0])



import numpy as np
from scipy import stats

# Sample data for purchase amounts in dollars
purchase_amounts = np.array([25.0, 30.0, 25.0, 40.0, 30.0, 50.0, 60.0, 30.0, 20.0, 25.0])

# Calculate the mean (average) purchase amount
mean_purchase_amount = np.mean(purchase_amounts)
print("Mean Purchase Amount:", mean_purchase_amount)

# Identify the mode of the purchase amounts
mode_result = stats.mode(purchase_amounts, keepdims=False)

# mode_result.mode should be a 1D array; access the first element correctly
print("Mode of Purchase Amounts:", mode_result.mode[0])

import numpy as np

# Sample data: rows represent departments, columns represent months
# Example: 3 departments, 12 months of data
expenses = np.array([
    [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100],  # Department 1
    [2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100],  # Department 2
    [1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600]   # Department 3
])

# Calculate the variance for each department (row-wise variance)
variance = np.var(expenses, axis=1)
print("Variance for Each Department:")
print(variance)

# Calculate the covariance matrix between departments
covariance_matrix = np.cov(expenses, rowvar=True)
print("\nCovariance Matrix:")
print(covariance_matrix)

import numpy as np

# Sample data: daily temperatures in degrees Celsius for one year (365 days)
# Replace this with your actual dataset
temperatures = np.array([
    12.5, 13.0, 13.5, 14.0, 14.5, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0,
    # ... (rest of the data)
    10.0, 10.5, 11.0, 11.5, 12.0  # Example values
])

# Calculate the variance of daily temperatures
variance = np.var(temperatures)
print("Variance of Daily Temperatures:", variance)

# Calculate the mean and standard deviation
mean_temp = np.mean(temperatures)
std_dev_temp = np.std(temperatures)

# Define a threshold to identify outliers (e.g., 2 standard deviations from the mean)
threshold = 2 * std_dev_temp

# Identify outliers
outliers = temperatures[(temperatures > mean_temp + threshold) | (temperatures < mean_temp - threshold)]
print("\nPotential Outliers:")
print(outliers)

import pandas as pd

# Sample data for daily temperatures of multiple cities
# Replace this with your actual dataset
data = {
    'City': ['New York', 'New York', 'New York', 'Los Angeles', 'Los Angeles', 'Los Angeles', 'Chicago', 'Chicago', 'Chicago'],
    'Temperature': [30, 32, 28, 75, 78, 80, 22, 25, 20]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Calculate mean temperature for each city
mean_temp = df.groupby('City')['Temperature'].mean()
print("Mean Temperature for Each City:")
print(mean_temp)

# Calculate standard deviation of temperature for each city
std_dev_temp = df.groupby('City')['Temperature'].std()
print("\nStandard Deviation of Temperature for Each City:")
print(std_dev_temp)

# Calculate the temperature range (difference between max and min temperatures) for each city
temp_range = df.groupby('City')['Temperature'].agg(lambda x: x.max() - x.min())
print("\nTemperature Range for Each City:")
print(temp_range)

# Determine the city with the highest temperature range
city_highest_range = temp_range.idxmax()
print("\nCity with the Highest Temperature Range:")
print(city_highest_range)

# Find the city with the most consistent temperature (lowest standard deviation)
city_most_consistent = std_dev_temp.idxmin()
print("\nCity with the Most Consistent Temperature:")
print(city_most_consistent)

import numpy as np

# Sample data: daily sales for the past month (30 days)
# Replace this with your actual sales data
daily_sales = np.array([
    200, 220, 210, 230, 240, 250, 260, 270, 280, 290,
    300, 310, 320, 330, 340, 350, 360, 370, 380, 390,
    400, 410, 420, 430, 440, 450, 460, 470, 480, 490
])

# Calculate the variance of daily sales
variance = np.var(daily_sales)
print("Variance of Daily Sales:", variance)

import pandas as pd
import numpy as np
from scipy import stats

# Sample data
data = {
    'product_title': ['Pineapple slicer', 'Levis Jeans Pant', 'Wallet', 'Salwar'],
    'product_category': ['Apparel', 'Apparel', 'Apparel', 'Apparel'],
    'star_rating': [4, 5, 5, 5],
    'review_headline': ['Really good', 'Perfect Dress', 'Love it', 'Awesome'],
    'review_date': ['2013-01-14', '2014-04-22', '2015-07-28', '2015-06-12']
}

# Create DataFrame
df = pd.DataFrame(data)

# Filter data for the specific product category
category = 'Apparel'
filtered_df = df[df['product_category'] == category]

# Calculate mean and standard deviation of the star ratings
ratings = filtered_df['star_rating']
mean_rating = ratings.mean()
std_dev_rating = ratings.std()
n = len(ratings)

# Calculate the 95% confidence interval for the mean rating
confidence = 0.95
z_score = stats.norm.ppf((1 + confidence) / 2)  # Z-score for 95% confidence
margin_of_error = z_score * (std_dev_rating / np.sqrt(n))
confidence_interval = (mean_rating - margin_of_error, mean_rating + margin_of_error)

# Print results
print("Mean Rating:", mean_rating)
print("Standard Deviation of Ratings:", std_dev_rating)
print("95% Confidence Interval for the Mean Rating:", confidence_interval)

import pandas as pd

# Sample data: diseases and the number of diagnosed patients
data = {
    'DISEASE_NAME': ['Common Cold', 'Diabetes', 'Bronchitis', 'Influenza', 'Kidney Stones'],
    'DIAGNOSED_PATIENTS': [320, 120, 100, 150, 60]
}

# Create DataFrame
df = pd.DataFrame(data)

# Calculate the frequency distribution (i.e., the count of patients for each disease)
# In this case, it's already provided as DIAGNOSED_PATIENTS

# Identify the most common disease
most_common_disease = df.loc[df['DIAGNOSED_PATIENTS'].idxmax()]

# Print the results
print("Frequency Distribution of Diseases:")
print(df)

print("\nMost Common Disease:")
print(most_common_disease)

import pandas as pd

# Sample data: weather conditions and the number of occurrences
data = {
    'WEATHER_CONDITION': ['Sunny', 'Rainy', 'Cloudy', 'Windy', 'Snowy'],
    'OCCURRENCES': [120, 80, 50, 30, 20]
}

# Create DataFrame
df = pd.DataFrame(data)

# Calculate the frequency distribution (i.e., the count of occurrences for each weather condition)
# In this case, it's already provided as OCCURRENCES

# Identify the most common weather condition
most_common_weather = df.loc[df['OCCURRENCES'].idxmax()]

# Print the results
print("Frequency Distribution of Weather Conditions:")
print(df)

print("\nMost Common Weather Condition:")
print(most_common_weather)

import pandas as pd

# Sample data: customer ages who made a purchase
data = {
    'customer_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'customer_age': [22, 34, 29, 45, 40, 33, 23, 31, 29, 28]
}

# Create DataFrame
df = pd.DataFrame(data)

# Calculate the frequency distribution of customer ages
age_frequency = df['customer_age'].value_counts().sort_index()

# Print the results
print("Frequency Distribution of Customer Ages:")
print(age_frequency)

import pandas as pd

# Sample data: number of likes for each post
data = {
    'post_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'likes': [120, 150, 200, 150, 100, 250, 300, 120, 200, 150]
}

# Create DataFrame
df = pd.DataFrame(data)

# Calculate the frequency distribution of likes
likes_frequency = df['likes'].value_counts().sort_index()

# Print the results
print("Frequency Distribution of Likes:")
print(likes_frequency)

import pandas as pd
from collections import Counter
import re
import string

# Sample data: customer reviews
data = {
    'review_id': [1, 2, 3, 4],
    'review_text': [
        'Great product, highly recommend!',
        'Not worth the price. The quality is poor.',
        'Absolutely love it! Will buy again.',
        'Quality is okay, but the service was excellent.'
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Function to preprocess text (convert to lowercase, remove punctuation, etc.)
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(f'[{string.punctuation}]', '', text)  # Remove punctuation
    return text

# Preprocess the reviews
df['cleaned_review'] = df['review_text'].apply(preprocess_text)

# Tokenize the words and calculate frequency distribution
all_words = ' '.join(df['cleaned_review']).split()
word_frequency = Counter(all_words)

# Convert the Counter to a DataFrame for easier display
word_frequency_df = pd.DataFrame(word_frequency.items(), columns=['Word', 'Frequency']).sort_values(by='Frequency', ascending=False)

# Print the results
print("Frequency Distribution of Words in Customer Reviews:")
print(word_frequency_df)







