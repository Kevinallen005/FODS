import numpy as np

# Create a sample NumPy array with fuel efficiency data (in miles per gallon)
fuel_efficiency = np.array([25.0, 30.0, 22.5, 27.5, 31.0, 29.5])

# Calculate the average fuel efficiency
average_fuel_efficiency = np.mean(fuel_efficiency)
print("Average Fuel Efficiency:", average_fuel_efficiency)

# Assume you want to compare the fuel efficiency of the first and last car models
fuel_efficiency_model1 = fuel_efficiency[0]  # Fuel efficiency of the first model
fuel_efficiency_model2 = fuel_efficiency[-1] # Fuel efficiency of the last model

# Calculate the percentage improvement
percentage_improvement = ((fuel_efficiency_model2 - fuel_efficiency_model1) / fuel_efficiency_model1) * 100
print("Percentage Improvement in Fuel Efficiency from Model 1 to Model 2:", percentage_improvement)
