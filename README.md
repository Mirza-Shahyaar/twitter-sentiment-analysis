# 🐦 Twitter Sentiment Analysis using Machine Learning

A machine learning project to classify Twitter posts as **positive** or **negative** using real tweet data from the [Sentiment140 dataset](https://www.kaggle.com/kazanova/sentiment140). The project demonstrates how to preprocess text data, build a sentiment classification model, and visualize public sentiment using word clouds.

---

## 📌 Project Objective

To build a sentiment analysis model that understands public sentiment on Twitter using natural language processing (NLP) and machine learning techniques.

---

## 💡 Features

- Preprocessing tweets: remove URLs, mentions, hashtags, and stopwords
- Text normalization using stemming
- Train a **Naive Bayes classifier**
- Predict sentiment (*positive* or *negative*) from tweet text
- Evaluate model performance using precision, recall, F1-score
- Visualize frequent words from positive tweets using a **word cloud**

---

## 🗂 Dataset

### 📦 Sentiment140 Twitter Dataset

- **Source:** [Sentiment140 on Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140)
- **Alternate Download:** [Direct CSV on GitHub](http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip)
- **Description:**
  - 1.6 million labeled tweets
  - Format: CSV with 6 columns (target, id, date, query, user, tweet)
  - Sentiment labels:  
    - `0` = Negative  
    - `4` = Positive  

> Place the file `training.1600000.processed.noemoticon.csv` in your project directory after downloading and unzipping.

---

## 🛠️ Technologies Used

- Python 3
- pandas, numpy
- NLTK (for text processing)
- scikit-learn (for ML model)
- matplotlib & WordCloud (for visualization)

---

## 🚀 How to Run

1. **Clone the repo:**

git clone https://github.com/your-username/twitter-sentiment-analysis.git
cd twitter-sentiment-analysis
Install requirements:

pip install -r requirements.txt
Download the dataset
Place training.1600000.processed.noemoticon.csv in the root folder.
You can download it from:

Kaggle
Or Direct ZIP
Run the notebook or script:
jupyter notebook sentiment_analysis.ipynb
# or
python sentiment_analysis.py
## 📊 Example Output
Model Accuracy: ~78% on test data

Sample word cloud from positive tweets:


## 🔍 Future Improvements
Add live Twitter scraping using Tweepy

Use TF-IDF for better text representation

Try advanced models like Logistic Regression, LSTM, or BERT

Build a real-time web app using Streamlit

## 🤝 Contributing
Contributions are welcome! Feel free to fork this repo and submit a pull request.

## 📄 License
This project is open-source and available under the MIT License.

