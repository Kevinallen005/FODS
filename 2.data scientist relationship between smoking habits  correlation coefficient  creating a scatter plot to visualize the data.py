import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create the sample dataset
data = {
    'IndividualID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'SmokingHabits': [20, 30, 10, 25, 5, 40, 35, 15, 50, 60],
    'LungCancer': [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
}

# Load the data into a DataFrame
df = pd.DataFrame(data)

# Display the DataFrame
print(df)

# Calculate the correlation coefficient
correlation_coefficient = np.corrcoef(df['SmokingHabits'], df['LungCancer'])[0, 1]
print(f"Correlation Coefficient: {correlation_coefficient}")

# Create a scatter plot
plt.scatter(df['SmokingHabits'], df['LungCancer'])
plt.title('Scatter Plot of Smoking Habits vs Lung Cancer')
plt.xlabel('Smoking Habits (Cigarettes per Day)')
plt.ylabel('Lung Cancer (1 = Yes, 0 = No)')
plt.show()
