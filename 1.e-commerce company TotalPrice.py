import pandas as pd

data = {
    'OrderID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'CustomerID': [1001, 1002, 1001, 1003, 1004, 1002, 1005, 1001, 1003, 1004],
    'ProductID': [2001, 2002, 2003, 2001, 2004, 2002, 2005, 2002, 2001, 2004],
    'Quantity': [2, 1, 3, 1, 2, 2, 1, 1, 4, 1],
    'TotalPrice': [50, 30, 45, 25, 40, 60, 20, 30, 100, 20]
}

df = pd.DataFrame(data)

print(df)
total_revenue = df['TotalPrice'].sum()
print(f"Total Revenue: ${total_revenue}")

orders_per_customer = df['CustomerID'].value_counts()
print("Number of Orders per Customer:")
print(orders_per_customer)

quantity_per_product = df.groupby('ProductID')['Quantity'].sum()
print("Total Quantity Sold per Product:")
print(quantity_per_product)

average_order_value = df['TotalPrice'].mean()
print(f"Average Order Value: ${average_order_value:.2f}")

revenue_per_customer = df.groupby('CustomerID')['TotalPrice'].sum().sort_values(ascending=False)
print("Top Customers by Revenue:")
print(revenue_per_customer)
