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
