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
