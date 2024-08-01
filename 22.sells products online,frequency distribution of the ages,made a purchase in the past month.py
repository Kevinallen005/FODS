import pandas as pd

# Sample data: customer ages who made a purchase
data = {
    'customer_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'customer_age': [22, 34, 29, 45, 40, 33, 23, 31, 29, 28]
}

# Create DataFrame
df = pd.DataFrame(data)

# Calculate the frequency distribution of customer ages
age_frequency = df['customer_age'].value_counts().sort_index()

# Print the results
print("Frequency Distribution of Customer Ages:")
print(age_frequency)
