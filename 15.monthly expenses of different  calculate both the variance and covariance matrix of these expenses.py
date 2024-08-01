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
