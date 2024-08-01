import pandas as pd

# Sample data: diseases and the number of diagnosed patients
data = {
    'DISEASE_NAME': ['Common Cold', 'Diabetes', 'Bronchitis', 'Influenza', 'Kidney Stones'],
    'DIAGNOSED_PATIENTS': [320, 120, 100, 150, 60]
}

# Create DataFrame
df = pd.DataFrame(data)

# Calculate the frequency distribution (i.e., the count of patients for each disease)
# In this case, it's already provided as DIAGNOSED_PATIENTS

# Identify the most common disease
most_common_disease = df.loc[df['DIAGNOSED_PATIENTS'].idxmax()]

# Print the results
print("Frequency Distribution of Diseases:")
print(df)

print("\nMost Common Disease:")
print(most_common_disease)
