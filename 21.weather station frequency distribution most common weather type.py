import pandas as pd

# Sample data: weather conditions and the number of occurrences
data = {
    'WEATHER_CONDITION': ['Sunny', 'Rainy', 'Cloudy', 'Windy', 'Snowy'],
    'OCCURRENCES': [120, 80, 50, 30, 20]
}

# Create DataFrame
df = pd.DataFrame(data)

# Calculate the frequency distribution (i.e., the count of occurrences for each weather condition)
# In this case, it's already provided as OCCURRENCES

# Identify the most common weather condition
most_common_weather = df.loc[df['OCCURRENCES'].idxmax()]

# Print the results
print("Frequency Distribution of Weather Conditions:")
print(df)

print("\nMost Common Weather Condition:")
print(most_common_weather)
