# 🧠 Dual-Model NLP Sentiment Engine

An end-to-end Machine Learning pipeline that processes raw product reviews and classifies their sentiment. This project physically demonstrates the performance gap between Traditional Statistical Machine Learning and Modern Deep Learning Transformer architecture.

## 🚀 Project Overview
This repository contains a complete MLOps lifecycle—from raw data cleaning to a live interactive web application. It pits a "bag-of-words" model against a state-of-the-art attention-based neural network to show why complex contextual language requires Deep Learning.

### The Two Models:
1. **The Baseline (Traditional ML):** TF-IDF Vectorizer + Logistic Regression.
   * *Accuracy:* 66.10%
   * *Weakness:* Fails to understand contextual negation (e.g., "Not bad") and defaults to a pessimistic bias on short sentences due to strict class balancing.
2. **The Heavyweight (Deep Learning):** Fine-Tuned DistilBERT (HuggingFace).
   * *Accuracy:* 86.62% (A 20.5% absolute improvement over the baseline).
   * *Strength:* Uses a bidirectional attention mechanism to understand sarcasm, double negatives, and subtle context.

## 🛠️ Tech Stack
* **Deep Learning:** PyTorch, HuggingFace `transformers` (DistilBERT)
* **Traditional ML:** `scikit-learn`
* **Data Engineering & EDA:** `pandas`, `nltk`, `beautifulsoup4`, `matplotlib`, `wordcloud`
* **Deployment & UI:** `streamlit` (with custom CSS styling)
* **Compute:** Google Colab (T4 GPU) for fine-tuning.

## 📊 The Architecture / Pipeline
1. **Data Engineering:** Built a custom cleaning pipeline to sanitize HTML tags, remove noisy stopwords, and specifically preserve negation words crucial for sentiment.
2. **Exploratory Data Analysis (EDA):** Visualized a 4:1 class imbalance (Positive:Negative) and generated word clouds to prove the necessity of advanced context parsing.
3. **Baseline Training:** Trained a Logistic Regression model locally, utilizing `class_weight='balanced'` to prevent the model from blindly guessing the majority class.
4. **Transformer Fine-Tuning:** Tokenized the dataset using `DistilBertTokenizerFast` and fine-tuned a pre-trained DistilBERT model over 3 epochs on a cloud GPU.
5. **Glass-Box UI:** Built a highly optimized Streamlit dashboard utilizing `@st.cache_resource` to pin the 250MB neural network to RAM for instant, real-time side-by-side inference.

## 💻 Installation & Local Usage

*Note: Due to GitHub's file size limits, the raw training dataset (286MB) and the fine-tuned BERT model weights (250MB) are not hosted in this repository.*

**1. Clone the repository:**
```bash
git clone [https://github.com/yogendra785/Dual_Model_Sentiment_Engine.git](https://github.com/yogendra785/Dual_Model_Sentiment_Engine.git)
cd Dual_Model_Sentiment_Engine
googl_drive_link:  
