import numpy as np

# Create the student_scores array
student_scores = np.array([
    [85, 90, 78, 92],
    [88, 76, 85, 85],
    [92, 88, 82, 89],
    [75, 85, 80, 87]
])

# Define the subject names
subjects = ['Math', 'Science', 'English', 'History']

# Calculate the average score for each subject
average_scores = student_scores.mean(axis=0)
print("Average Scores for Each Subject:", average_scores)

# Identify the subject with the highest average score
highest_avg_score_index = np.argmax(average_scores)
highest_avg_score_subject = subjects[highest_avg_score_index]
print("Subject with the Highest Average Score:", highest_avg_score_subject)
