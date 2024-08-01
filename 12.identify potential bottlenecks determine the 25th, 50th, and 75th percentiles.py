import numpy as np

# Sample data for response times in milliseconds
response_times = np.array([120, 150, 180, 200, 220, 240, 260, 280, 300, 320])

# Calculate the 25th, 50th, and 75th percentiles
percentiles = np.percentile(response_times, [25, 50, 75])
print("Percentiles using NumPy:")
print("25th percentile:", percentiles[0])
print("50th percentile (Median):", percentiles[1])
print("75th percentile:", percentiles[2])
