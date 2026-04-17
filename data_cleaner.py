import pandas as pd
import re
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words=set(stopwords.words('english'))

stop_words.discard('not')
stop_words.discard('no')

def clean_text(text):
    """
    The master Cleaning Function.
    Takes in a review and returns a machine-readable string.
    """
    if pd.isna(text):
        return ""
    text=str(text)

    text=BeautifulSoup(text, "html.parser").get_text()
    #removing URL
    text=re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    text=text.lower()

    words=text.split()
    cleaned_words=[word for word in words if word not in stop_words]
    return " ".join(cleaned_words)

if __name__ == "__main__":
    print("Booting up the data cleaning pipeline...")
    df=pd.read_csv('data/raw_reviews.csv')
    df=df.sample(10000, random_state=42).reset_index(drop=True)
    print(f"Processing {len(df)} rows. This might take few minutes...")
    df['cleaned_text']=df['Text'].apply(clean_text)
    df['sentiment']=df['Score'].apply(lambda x:1 if x>3 else 0)
    final_df=df[['cleaned_text','sentiment']]
    final_df.to_csv('data/cleaned_reviews.csv',index=False)
    print("Pipeline Completed! saved to 'data/cleaned_reviews.csv'")

