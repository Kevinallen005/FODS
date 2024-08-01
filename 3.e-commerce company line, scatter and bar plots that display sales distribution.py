import pandas as pd
import matplotlib.pyplot as plt

# Create the sample dataset
data = {
    'Month': ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'],
    'Electronics': [2000, 1800, 2200, 2100, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000],
    'Fashion': [1500, 1600, 1400, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500],
    'Home': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100],
    'Books': [500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600],
    'Toys': [300, 400, 350, 450, 500, 550, 600, 650, 700, 750, 800, 850]
}

# Load the data into a DataFrame
df = pd.DataFrame(data)

# Plotting the data
months = df['Month']
categories = ['Electronics', 'Fashion', 'Home', 'Books', 'Toys']

# Line plot for each category
plt.figure(figsize=(10, 6))
for category in categories:
    plt.plot(months, df[category], label=category)
plt.title('Monthly Sales by Product Category')
plt.xlabel('Month')
plt.xticks(rotation=45)
plt.ylabel('Sales')
plt.legend()
plt.show()

# Scatter plot for each category
plt.figure(figsize=(10, 6))
for category in categories:
    plt.scatter(months, df[category], label=category)
plt.title('Monthly Sales by Product Category')
plt.xlabel('Month')
plt.xticks(rotation=45)
plt.ylabel('Sales')
plt.legend()
plt.show()

# Bar plot for each category
plt.figure(figsize=(10, 6))
bar_width = 0.15
index = range(len(months))

for i, category in enumerate(categories):
    plt.bar([p + bar_width*i for p in index], df[category], width=bar_width, label=category)

plt.title('Monthly Sales by Product Category')
plt.xlabel('Month')
plt.xticks([p + bar_width*2 for p in index], months, rotation=45)
plt.ylabel('Sales')
plt.legend()
plt.show()
