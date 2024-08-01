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
