import pandas as pd

# Sample data: number of likes for each post
data = {
    'post_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'likes': [120, 150, 200, 150, 100, 250, 300, 120, 200, 150]
}

# Create DataFrame
df = pd.DataFrame(data)

# Calculate the frequency distribution of likes
likes_frequency = df['likes'].value_counts().sort_index()

# Print the results
print("Frequency Distribution of Likes:")
print(likes_frequency)
