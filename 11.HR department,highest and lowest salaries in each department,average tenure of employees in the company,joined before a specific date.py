import pandas as pd

# Sample data for demonstration purposes
data = {
    'EmployeeID': [101, 102, 103, 104, 105],
    'Department': ['HR', 'Finance', 'IT', 'HR', 'IT'],
    'Salary': [50000, 60000, 55000, 52000, 58000],
    'JoiningDate': ['2021-06-01', '2020-03-15', '2022-01-10', '2021-11-25', '2021-05-30']
}

# Create a DataFrame from the sample data
df = pd.DataFrame(data)

# Convert 'JoiningDate' column to datetime
df['JoiningDate'] = pd.to_datetime(df['JoiningDate'])

# 1. Determine the highest and lowest salaries in each department
salary_stats = df.groupby('Department')['Salary'].agg(['max', 'min'])
print("\nHighest and Lowest Salaries in Each Department:")
print(salary_stats)

# 2. Calculate the average tenure of employees in the company
# Current date
current_date = pd.to_datetime('today')

# Calculate tenure in days
df['Tenure'] = (current_date - df['JoiningDate']).dt.days

# Calculate average tenure
average_tenure = df['Tenure'].mean()
print("\nAverage Tenure of Employees in Days:", average_tenure)

# 3. Identify employees who joined before a specific date
specific_date = pd.to_datetime('2022-01-01')
employees_before_date = df[df['JoiningDate'] < specific_date]
print(f"\nEmployees who joined before {specific_date.date()}:")
print(employees_before_date[['EmployeeID', 'Department', 'JoiningDate']])
