
import streamlit as st
import pickle
import os


# Project root folder path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# Load model and vectorizer
model_path = os.path.join(BASE_DIR, "models", "logistic_regression_model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")

model = pickle.load(open(model_path, "rb"))
vectorizer = pickle.load(open(vectorizer_path, "rb"))
print(type(vectorizer))
print(hasattr(vectorizer, "idf_"))

st.sidebar.title("📌 Project Info")

st.sidebar.markdown("""
**Project By:** Abhilasha Mishra

**Model:** Logistic Regression

**Vectorizer:** TF-IDF

**Technology:** Machine Learning + Streamlit

**Purpose:** Detect Fake and Real News
""")
# Streamlit UI
st.title("📰 TruthLens AI")

st.markdown("""
### AI-powered News Verification System
Enter a news headline or article text to check whether it is likely Real or Fake.
""")

news = st.text_area(
    "Enter News Text :",
    placeholder="Paste a news headline or article here..."
)

if st.button("🔍 Check News"):
    if news.strip() == "":
        st.warning("Please enter some news text.")
    else:
        news_vectorized = vectorizer.transform([news])
        prediction = model.predict(news_vectorized)
        probability = model.predict_proba(news_vectorized)
        confidence = max(probability[0]) * 100

        if prediction[0] == 1:
            st.success("✅ REAL NEWS")
        else:
            st.error("❌ FAKE NEWS")
        
        st.progress(int(confidence))
        st.metric("Confidence Score", f"{confidence:.2f}%")
        st.caption(
        "Note: This model predicts based on patterns learned from historical training data and may not verify recent real-world events."
    )