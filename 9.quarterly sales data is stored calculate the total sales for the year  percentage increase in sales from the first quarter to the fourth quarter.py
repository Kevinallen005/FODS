import numpy as np

# Create a sample NumPy array with quarterly sales data
sales_data = np.array([15000, 20000, 25000, 30000])

# Calculate the total sales for the year
total_sales = np.sum(sales_data)
print("Total Sales for the Year:", total_sales)

# Calculate the percentage increase from the first quarter to the fourth quarter
sales_q1 = sales_data[0]  # Sales in the first quarter
sales_q4 = sales_data[3]  # Sales in the fourth quarter

percentage_increase = ((sales_q4 - sales_q1) / sales_q1) * 100
print("Percentage Increase from Q1 to Q4:", percentage_increase)
