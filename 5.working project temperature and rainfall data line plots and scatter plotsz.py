import pandas as pd
import matplotlib.pyplot as plt

# Create the sample dataset
data = {
    'Month': ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'],
    'Temperature': [30, 32, 35, 40, 45, 50, 55, 54, 48, 42, 35, 32],
    'Rainfall': [50, 40, 45, 30, 20, 10, 5, 8, 15, 25, 35, 40]
}

# Load the data into a DataFrame
df = pd.DataFrame(data)

# Display the DataFrame
print(df)

# Line plot for temperature and rainfall
plt.figure(figsize=(10, 6))
plt.plot(df['Month'], df['Temperature'], marker='o', label='Temperature (°C)')
plt.plot(df['Month'], df['Rainfall'], marker='o', label='Rainfall (mm)')
plt.title('Monthly Temperature and Rainfall')
plt.xlabel('Month')
plt.xticks(rotation=45)
plt.ylabel('Value')
plt.legend()
plt.show()

# Scatter plot for temperature and rainfall
plt.figure(figsize=(10, 6))
plt.scatter(df['Month'], df['Temperature'], label='Temperature (°C)')
plt.scatter(df['Month'], df['Rainfall'], label='Rainfall (mm)')
plt.title('Monthly Temperature and Rainfall')
plt.xlabel('Month')
plt.xticks(rotation=45)
plt.ylabel('Value')
plt.legend()
plt.show()
