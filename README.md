# Fake News Detector 

This is a simple Machine Learning project that predicts whether a given news statement is Real or Fake.

The project is built using Python, Scikit-Learn and Streamlit. It uses TF-IDF Vectorization for converting text into numerical features and Logistic Regression for classification.

## Features

- Detects Real and Fake News
- User-friendly Streamlit interface
- Confidence score display
- TF-IDF based text processing
- Logistic Regression classification model

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Streamlit
- Pickle

## Project Structure

FAKE_NEWS_DETECTOR

├── data

│   └── news.csv

├── models

│   ├── logistic_regression_model.pkl

│   └── tfidf_vectorizer.pkl

├── notebook

│   ├── app

│   │   └── app.py

│   └── eda.ipynb

└── README.md

## How It Works

1. User enters a news statement.
2. The text is converted into TF-IDF features.
3. The trained Logistic Regression model makes a prediction.
4. The application displays whether the news is Real or Fake along with a confidence score.

## Limitation

This model predicts based on patterns learned from the training dataset. It does not have access to current real-world events or live news verification.

## Future Improvements

- Use larger and more recent datasets
- Try advanced models such as Random Forest or BERT
- Add live news verification using APIs
- Improve prediction accuracy

## Project By

Abhilasha Mishra

B.Tech CSE (AI)