import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pickle

print("Booting up the Baseline ML Pipeline....")

df=pd.read_csv('data/cleaned_reviews.csv')
df=df.dropna(subset=['cleaned_text'])

X=df['cleaned_text']
y=df['sentiment']

X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2,random_state=42)
print(f"Training on {len(X_train)} reviews, Testing on {len(X_test)} reviews.")

#TF-IDF Vectorization
print("Converting text to numbers using TF-IDF...")

vectorizer=TfidfVectorizer(max_features=5000)
#Fit on training data and then transforming for train and test data
X_train_vec=vectorizer.fit_transform(X_train)
X_test_vec=vectorizer.fit_transform(X_test)

print("Training Logistic Regression Model...")

model=LogisticRegression(class_weight='balanced',random_state=42)
model.fit(X_train_vec,y_train)

#Evaluating the model
print("\n Model Evaluation on Unseen Test Data:")
y_pred=model.predict(X_test_vec)
accuracy=accuracy_score(y_test,y_pred)
print(f"Total Accuracy : {accuracy*100:.2f}%\n")
print("Detailed report")
print(classification_report(y_test,y_pred))

#saving the model and vectorizer
print("\n Saving model weights to the /models folder...")
with open('models/tfidf_vectorizer.pkl','wb') as f:
    pickle.dump(vectorizer, f)
with open('models/logistic_model.pkl','wb') as f:
    pickle.dump(model, f)

print("✅ Baseline Training Complete! Ready for Deployment.")

