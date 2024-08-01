import pandas as pd
from collections import Counter
import re
import string

# Sample data: customer reviews
data = {
    'review_id': [1, 2, 3, 4],
    'review_text': [
        'Great product, highly recommend!',
        'Not worth the price. The quality is poor.',
        'Absolutely love it! Will buy again.',
        'Quality is okay, but the service was excellent.'
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Function to preprocess text (convert to lowercase, remove punctuation, etc.)
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(f'[{string.punctuation}]', '', text)  # Remove punctuation
    return text

# Preprocess the reviews
df['cleaned_review'] = df['review_text'].apply(preprocess_text)

# Tokenize the words and calculate frequency distribution
all_words = ' '.join(df['cleaned_review']).split()
word_frequency = Counter(all_words)

# Convert the Counter to a DataFrame for easier display
word_frequency_df = pd.DataFrame(word_frequency.items(), columns=['Word', 'Frequency']).sort_values(by='Frequency', ascending=False)

# Print the results
print("Frequency Distribution of Words in Customer Reviews:")
print(word_frequency_df)
