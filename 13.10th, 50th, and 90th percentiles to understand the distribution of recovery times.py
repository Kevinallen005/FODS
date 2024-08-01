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
