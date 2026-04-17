import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import wordcloud, WordCloud

print("Booting up Exploratory Data Analysis...")

df=pd.read_csv('data/cleaned_reviews.csv')
df=df.dropna(subset=['cleaned_text'])

print("Generating Sentiment Distribution Chart...")

sns.countplot(x='sentiment', data=df, palette='Set2')
plt.title("Sentiment Distribution (0 = Negative, 1 = Positive)")
plt.xlabel("Sentiment Score")
plt.ylabel("Number of Reviews")
plt.show()

# --- Visual 2: Word Clouds ---
print("☁️ Generating Word Clouds...")


positive_text=" ".join(df[df['sentiment']==1]['cleaned_text'].astype(str))
negative_text = " ".join(df[df['sentiment'] == 0]['cleaned_text'].astype(str))

#positive word cloud

plt.figure(figsize=(12,6))
wordcloud_pos = WordCloud(width=800,height=400,background_color='white',colormap='Greens').generate(positive_text)
plt.imshow(wordcloud_pos,interpolation='bilinear')
plt.title("Most Common words in POSITIVE Reviews",fontsize=16)
plt.axis("off")
plt.show()

# Create the Negative Word Cloud
plt.figure(figsize=(12, 6))
wordcloud_neg = WordCloud(width=800, height=400, background_color='black', colormap='Reds').generate(negative_text)
plt.imshow(wordcloud_neg, interpolation='bilinear')
plt.title("Most Common Words in NEGATIVE Reviews", fontsize=16, color='white')
plt.axis("off")
plt.show()

print("✅ EDA Complete. Look at the charts to understand your data!")