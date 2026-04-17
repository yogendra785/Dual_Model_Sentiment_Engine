# 🧠 Dual-Model NLP Sentiment Engine [IN PROGRESS]

An end-to-end Natural Language Processing pipeline comparing Traditional Machine Learning with Deep Learning Transformer architecture. 

## 🚀 Project Overview
This project processes raw, noisy product reviews and classifies their sentiment. It is built in two layers to demonstrate the value of Deep Learning over traditional statistical methods:
1. **Baseline Model:** TF-IDF + Logistic Regression
2. **Deep Learning Model:** Fine-Tuned DistilBERT (HuggingFace)
3. **Deployment:** Streamlit UI (Upcoming)

## 📊 Current Progress
* **Phase 1 & 2 (Data Engineering):** Built a custom cleaning pipeline using BeautifulSoup and NLTK to handle HTML tags, stopwords, and negation preservation.
* **Phase 3 (EDA):** Generated sentiment distribution and class-separated word clouds. Identified class imbalance.
* **Phase 4 (Baseline ML):** Trained a Logistic Regression model with `class_weight='balanced'`. 
  * *Baseline Accuracy:* 66.10% (Struggles with complex contextual negation, proving the need for BERT).

## 🛠️ Tech Stack So Far
* `pandas`, `scikit-learn`, `nltk`, `beautifulsoup4`, `matplotlib`, `wordcloud`

*(Note: Data and model weights are excluded via .gitignore due to size constraints).*