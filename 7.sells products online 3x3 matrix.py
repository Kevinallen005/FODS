import numpy as np

# Example 3x3 NumPy array representing sales data for three products
# Each row represents a different product and each element represents a sale price
sales_data = np.array([[100, 200, 300],
                       [150, 250, 350],
                       [180, 280, 380]])

# Calculate the average price of all products sold
average_price = np.mean(sales_data)

print(f"The average price of all products sold in the past month is: ${average_price:.2f}")
