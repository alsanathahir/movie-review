import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import io
import joblib
import logging
import chardet  # For detecting file encoding

# Set up logging
logging.basicConfig(level=logging.INFO, filename='app.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Define a flexible color palette
SENTIMENT_COLORS = {'positive': '#66c2a5', 'negative': '#fc8d62', 'neutral': '#8da0cb'}

def preprocess_text(text):
    """Clean and preprocess text data."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    return text

def detect_columns(df):
    """Auto-detect text and label columns."""
    text_candidates = [col for col in df.columns if any(keyword in col.lower() for keyword in ['text', 'review', 'comment'])]
    label_candidates = [col for col in df.columns if any(keyword in col.lower() for keyword in ['sentiment', 'label', 'class'])]

    text_column = text_candidates[0] if text_candidates else [col for col in df.columns if df[col].dtype == 'object'][0]
    label_column = label_candidates[0] if label_candidates else None
    return text_column, label_column

def detect_encoding(file):
    """Detect file encoding."""
    raw_data = file.read()
    result = chardet.detect(raw_data)
    file.seek(0)  # Reset file pointer
    return result['encoding']

def load_file(uploaded_file):
    """Load file with encoding detection."""
    try:
        if uploaded_file.name.endswith('.csv'):
            encoding = detect_encoding(uploaded_file)
            return pd.read_csv(uploaded_file, encoding=encoding, on_bad_lines='skip')
        elif uploaded_file.name.endswith('.xlsx') or uploaded_file.name.endswith('.xls'):
            return pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.txt'):
            encoding = detect_encoding(uploaded_file)
            return pd.read_csv(uploaded_file, delimiter='\t', encoding=encoding, on_bad_lines='skip')
        else:
            st.error("Unsupported file format. Please upload CSV, Excel, TXT.")
            return None
    except Exception as e:
        st.error(f"Error loading file: {e}")
        logging.error(f"Error loading file: {e}")
        return None

@st.cache_resource
def train_sentiment_model(df_train, text_column, label_column):
    """Train a sentiment model."""
    df_train = df_train.dropna(subset=[text_column, label_column])  # Remove nulls in text and label columns
    df_train[text_column] = df_train[text_column].apply(preprocess_text)

    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df_train[text_column])
    y = df_train[label_column]

    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    joblib.dump(model, 'sentiment_model.pkl')
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
    return model, vectorizer, accuracy, precision, recall, f1

def load_model_and_vectorizer():
    """Load the trained model and vectorizer."""
    try:
        model = joblib.load('sentiment_model.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
    except FileNotFoundError:
        return None, None
    return model, vectorizer

def predict_sentiment(texts, model, vectorizer):
    """Predict sentiment."""
    processed_texts = [preprocess_text(text) for text in texts]
    X = vectorizer.transform(processed_texts)
    predictions = model.predict(X)
    probs = model.predict_proba(X)
    return predictions, probs

def plot_sentiment_distribution(df, label_column, title="Sentiment Distribution"):
    """Plot sentiment distribution."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(x=df[label_column], ax=ax, palette=SENTIMENT_COLORS, hue=df[label_column], legend=False)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Sentiment", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

def plot_word_cloud(df, text_column, label_column, sentiment):
    """Plot word cloud for a specific sentiment."""
    words = ' '.join(df[df[label_column] == sentiment][text_column].dropna())
    if words:
        wordcloud = WordCloud(width=800, height=400, background_color='white',
                              stopwords=set(string.punctuation)).generate(words)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.set_title(f'{sentiment.capitalize()} Words', fontsize=14)
        ax.axis('off')
        st.pyplot(fig)
        plt.close()

def plot_text_length_distribution(df, text_column):
    """Plot text length distribution."""
    df['text_length'] = df[text_column].astype(str).apply(len)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(df['text_length'], bins=30, kde=True, ax=ax, color='#8da0cb')
    ax.set_title("Text Length Distribution", fontsize=14)
    ax.set_xlabel("Text Length (characters)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    ax.set_title("Confusion Matrix", fontsize=14)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

def display_sentiment_counts(df, label_column, title="Sentiment Counts"):
    """Display sentiment counts."""
    counts = df[label_column].value_counts()
    st.write(f"**{title}:**")
    for sentiment, count in counts.items():
        st.write(f"{sentiment.capitalize()}: {count}")

def main():
    """Main function to run the Streamlit app."""
    st.title("🎬 Movie Review Sentiment Classifier")
    st.markdown("""
    Analyze sentiments in movie reviews. Upload your dataset, explore it, train a model,
    and predict sentiments with detailed metrics and visualizations.
    """)

    # Step 1: Upload & Preview Dataset
    st.header("1. Upload & Preview Dataset 📂")
    uploaded_file = st.file_uploader("Upload dataset (CSV, TXT, XLS, XLSX)", type=["csv", "txt", "xls", "xlsx"])

    if uploaded_file:
        df = load_file(uploaded_file)
        if df is None or df.empty:
            st.error("Failed to load dataset or dataset is empty.")
            return

        st.write("### Data Preview:")
        st.dataframe(df.head())

        # Auto-detect columns
        text_column, label_column = detect_columns(df)
        st.write(f"**Auto-Detected Columns:** Text: {text_column}, Label: {label_column}")

        # Allow manual override
        text_column = st.selectbox("Override Text Column:", df.columns, index=df.columns.get_loc(text_column))
        label_column = st.selectbox("Override Label Column:", df.columns, index=df.columns.get_loc(label_column))

        # Step 2: Exploratory Data Analysis (EDA)
        st.header("2. Exploratory Data Analysis (EDA) 🕵️‍♂️")
        st.write("### Dataset Summary:")
        st.write(df.describe(include="all"))

        st.write("### Missing Values:")
        st.write(df.isnull().sum())

        display_sentiment_counts(df, label_column, "Sentiment Counts Before Prediction")
        plot_sentiment_distribution(df, label_column, "Sentiment Distribution Before Prediction")

        st.write("### Word Clouds:")
        col1, col2 = st.columns(2)
        with col1:
            plot_word_cloud(df, text_column, label_column, 'positive')
        with col2:
            plot_word_cloud(df, text_column, label_column, 'negative')

        st.write("### Text Length Distribution:")
        plot_text_length_distribution(df, text_column)

        # Step 3: Data Cleaning & Preprocessing
        st.header("3. Data Cleaning & Preprocessing 🛠️")
        df[text_column] = df[text_column].apply(preprocess_text)

        if st.checkbox("Show Cleaned Data Preview"):
            st.write("**Cleaned Data Preview:**")
            st.write(df[[text_column]].head())

        # Step 4: Train & Evaluate a Sentiment Model
        st.header("4. Train & Evaluate Sentiment Model 📈")
        model, vectorizer = load_model_and_vectorizer()

        if st.button("Train Model"):
            model, vectorizer, accuracy, precision, recall, f1 = train_sentiment_model(df, text_column, label_column)
            if model is None:
                return
            st.success(f"Model trained successfully! 🎉")

            # Display model metrics
            st.write("### Model Performance on Test Set:")
            st.write(f"**Accuracy:** {accuracy:.2f}")
            st.write(f"**Precision (weighted):** {precision:.2f}")
            st.write(f"**Recall (weighted):** {recall:.2f}")
            st.write(f"**F1-Score (weighted):** {f1:.2f}")

            # Predict on full dataset
            predictions, probs = predict_sentiment(df[text_column], model, vectorizer)
            df['predicted_sentiment'] = predictions
            st.write("### Predictions on Uploaded Data:")
            st.write(df[[text_column, 'predicted_sentiment']].head())

            # Sentiment counts after prediction
            display_sentiment_counts(df, 'predicted_sentiment', "Sentiment Counts After Prediction")
            plot_sentiment_distribution(df, 'predicted_sentiment', "Sentiment Distribution After Prediction")

            # Full dataset metrics
            y_true = df[label_column]
            y_pred = df['predicted_sentiment']
            full_accuracy = accuracy_score(y_true, y_pred)
            full_precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            full_recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            full_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            st.write("### Model Performance on Full Dataset:")
            st.write(f"**Accuracy:** {full_accuracy:.2f}")
            st.write(f"**Precision (weighted):** {full_precision:.2f}")
            st.write(f"**Recall (weighted):** {full_recall:.2f}")
            st.write(f"**F1-Score (weighted):** {full_f1:.2f}")
            st.text(classification_report(y_true, y_pred))
            plot_confusion_matrix(y_true, y_pred, model.classes_)

        # Load existing model if no training
        elif model:
            predictions, probs = predict_sentiment(df[text_column], model, vectorizer)
            df['predicted_sentiment'] = predictions
            st.write("### Predictions Using Pre-Trained Model:")
            st.write(df[[text_column, 'predicted_sentiment']].head())
            display_sentiment_counts(df, 'predicted_sentiment', "Sentiment Counts After Prediction")
            plot_sentiment_distribution(df, 'predicted_sentiment', "Sentiment Distribution After Prediction")

        # Step 5: Sentiment Prediction on New Text
        st.header("5. Predict Sentiment on New Review 🌟")
        user_input = st.text_area("Enter a movie review:")
        if st.button("Predict Sentiment") and user_input and model:
            pred, prob = predict_sentiment([user_input], model, vectorizer)
            sentiment = pred[0]
            prob_dict = dict(zip(model.classes_, prob[0]))
            st.write(f"**Predicted Sentiment:** {sentiment.capitalize()}")
            st.write(f"**Probabilities:** {prob_dict}")

        # Download Results
        st.header("6. Download Processed Data 📥")
        output = io.BytesIO()
        df_to_download = df[[text_column, 'predicted_sentiment']] if 'predicted_sentiment' in df.columns else df[[text_column]]
        df_to_download.to_csv(output, index=False)
        output.seek(0)
        st.download_button(
            label="Download Processed Data",
            data=output,
            file_name="processed_sentiment_data.csv",
            mime="text/csv"
        )

        # Documentation
        st.header("How It Works 📘")
        st.markdown("""
        - **Data Handling**: Upload datasets (CSV, Excel, TXT) with movie reviews and sentiment labels.
        - **Cleaning**: Text is normalized and tokenized.
        - **Model**: Logistic Regression with TF-IDF features, trained locally.
        - **Prediction**: Analyzes new reviews with a confidence threshold for clear sentiment.
        - **Results**: View counts, distributions, metrics (accuracy, precision, recall, F1), and word clouds.
        """)

if __name__ == "__main__":
    main()
