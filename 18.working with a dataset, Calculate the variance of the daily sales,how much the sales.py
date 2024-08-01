import numpy as np

# Sample data: daily sales for the past month (30 days)
# Replace this with your actual sales data
daily_sales = np.array([
    200, 220, 210, 230, 240, 250, 260, 270, 280, 290,
    300, 310, 320, 330, 340, 350, 360, 370, 380, 390,
    400, 410, 420, 430, 440, 450, 460, 470, 480, 490
])

# Calculate the variance of daily sales
variance = np.var(daily_sales)
print("Variance of Daily Sales:", variance)
