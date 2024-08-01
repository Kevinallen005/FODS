import pandas as pd
import matplotlib.pyplot as plt

# Create the sample dataset
data = {
    'Month': ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'],
    'Sales': [2000, 2200, 2500, 2400, 2600, 2700, 3000, 3100, 2900, 2800, 3200, 3300]
}

# Load the data into a DataFrame
df = pd.DataFrame(data)

# Display the DataFrame
print(df)

# Line plot for monthly sales data
plt.figure(figsize=(10, 6))
plt.plot(df['Month'], df['Sales'], marker='o', linestyle='-', color='b')
plt.title('Monthly Sales Data')
plt.xlabel('Month')
plt.xticks(rotation=45)
plt.ylabel('Sales')
plt.grid(True)
plt.show()
import pandas as pd
import matplotlib.pyplot as plt

# Create the sample dataset
data = {
    'Month': ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'],
    'Sales': [2000, 2200, 2500, 2400, 2600, 2700, 3000, 3100, 2900, 2800, 3200, 3300]
}

# Load the data into a DataFrame
df = pd.DataFrame(data)

# Display the DataFrame
print(df)

# Bar plot for monthly sales data
plt.figure(figsize=(10, 6))
plt.bar(df['Month'], df['Sales'], color='c')
plt.title('Monthly Sales Data')
plt.xlabel('Month')
plt.xticks(rotation=45)
plt.ylabel('Sales')
plt.grid(axis='y')
plt.show()
