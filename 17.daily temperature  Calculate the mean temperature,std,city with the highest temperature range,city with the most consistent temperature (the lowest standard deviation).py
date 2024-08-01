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
