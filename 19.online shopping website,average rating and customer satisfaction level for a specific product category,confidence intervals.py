import pandas as pd
import numpy as np
from scipy import stats

# Sample data
data = {
    'product_title': ['Pineapple slicer', 'Levis Jeans Pant', 'Wallet', 'Salwar'],
    'product_category': ['Apparel', 'Apparel', 'Apparel', 'Apparel'],
    'star_rating': [4, 5, 5, 5],
    'review_headline': ['Really good', 'Perfect Dress', 'Love it', 'Awesome'],
    'review_date': ['2013-01-14', '2014-04-22', '2015-07-28', '2015-06-12']
}

# Create DataFrame
df = pd.DataFrame(data)

# Filter data for the specific product category
category = 'Apparel'
filtered_df = df[df['product_category'] == category]

# Calculate mean and standard deviation of the star ratings
ratings = filtered_df['star_rating']
mean_rating = ratings.mean()
std_dev_rating = ratings.std()
n = len(ratings)

# Calculate the 95% confidence interval for the mean rating
confidence = 0.95
z_score = stats.norm.ppf((1 + confidence) / 2)  # Z-score for 95% confidence
margin_of_error = z_score * (std_dev_rating / np.sqrt(n))
confidence_interval = (mean_rating - margin_of_error, mean_rating + margin_of_error)

# Print results
print("Mean Rating:", mean_rating)
print("Standard Deviation of Ratings:", std_dev_rating)
print("95% Confidence Interval for the Mean Rating:", confidence_interval)
