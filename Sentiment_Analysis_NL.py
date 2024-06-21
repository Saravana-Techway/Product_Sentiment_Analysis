import streamlit as st
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Print the current working directory
print("Current working directory:", os.getcwd())

# Step 1: Load CSV files
csv_files = [
    'MI 10T - 128GB_reviews.csv',
    'Micromax Dual 4  64 GB_reviews.csv',
    'Motorola Edge 50 Pro 5G - 256GB_reviews.csv',
    'Vivo V23e 5G - 128GB_reviews.csv',
    'Oneplus 10R - 128GB_reviews.csv',
    'Samsung Galaxy S23 FE- 128GB_reviews.csv'
]

def get_product_name(df):
    return df.iloc[0, 0]  # First row, first column

dataframes = {}
for file in csv_files:
    try:
        df = pd.read_csv(file)
        product_name = get_product_name(df)
        dataframes[product_name] = df
    except FileNotFoundError:
        print(f"File not found: {file}")
    except Exception as e:
        print(f"Error loading {file}: {e}")

# Sentiment Analysis Function
def get_sentiment(review):
    review = str(review)
    blob = TextBlob(review)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    if polarity > 0:
        sentiment_def = "Positive"
    elif polarity < 0:
        sentiment_def = "Negative"
    else:
        sentiment_def = "Neutral"
    return polarity, subjectivity, sentiment_def

def aggregate_sentiments(reviews):
    polarities = []
    subjectivities = []
    sentiment_counts = {'Positive': 0, 'Negative': 0, 'Neutral': 0}

    for review in reviews:
        polarity, subjectivity, sentiment_name = get_sentiment(review)
        polarities.append(polarity)
        subjectivities.append(subjectivity)
        sentiment_counts[sentiment_name] += 1

    avg_polarity = sum(polarities) / len(polarities) if polarities else 0
    avg_subjectivity = sum(subjectivities) / len(subjectivities) if subjectivities else 0
    overall_sentiment = max(sentiment_counts, key=sentiment_counts.get)

    return avg_polarity, avg_subjectivity, overall_sentiment, sentiment_counts


def interpret_subjectivity(subjectivity):
    if subjectivity < 0.2:
        return "Very Objective"
    elif 0.2 <= subjectivity < 0.4:
        return "Objective"
    elif 0.4 <= subjectivity < 0.6:
        return "Neutral"
    elif 0.6 <= subjectivity < 0.8:
        return "Subjective"
    else:
        return "Very Subjective"

def compare_products(query):
    sentiments = {}
    doc = nlp(query)
    query_keywords = [token.text for token in doc if token.is_alpha]

    # Initialize TfidfVectorizer and fit to all reviews
    vectorizer = TfidfVectorizer().fit([review for df in dataframes.values() for review in df['Review'].tolist()])

    for product, df in dataframes.items():
        reviews = df['Review'].tolist()  # Assuming 'Review' is the column containing reviews

        # Transform reviews into TF-IDF vectors
        reviews_transformed = vectorizer.transform(reviews)

        # Transform query into TF-IDF vector
        query_vec = vectorizer.transform([query])

        # Calculate cosine similarities between query and reviews
        cosine_similarities = cosine_similarity(query_vec, reviews_transformed).flatten()

        # Filter reviews based on cosine similarity
        threshold = 0.1  # Adjust threshold as needed
        filtered_reviews = [reviews[i] for i in range(len(reviews)) if cosine_similarities[i] > threshold]

        if filtered_reviews:
            try:
                avg_polarity, avg_subjectivity, overall_sentiment, sentiment_counts = aggregate_sentiments(filtered_reviews)
                sentiments[product] = {
                    'average_polarity': avg_polarity,
                    'average_subjectivity': avg_subjectivity,
                    'overall_sentiment': overall_sentiment,
                    'Positive_Count': sentiment_counts["Positive"],
                    'Negative_Count': sentiment_counts["Negative"],
                    'Neutral_Count': sentiment_counts["Neutral"]
                }
            except Exception as e:
                st.error(f"Error processing {product}: {e}")

    return sentiments

# Step 5: Streamlit App
def main():
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/ed/Pandas_logo.svg/1200px-Pandas_logo.svg.png",
             use_column_width=True)
    st.title('Product Sentiment Analysis')

    # User input
    user_query = st.text_input('Enter your query:')
    if user_query:
        # Perform sentiment analysis on the user query
        query_sentiment = get_sentiment(user_query)
        query_polarity = query_sentiment[0]  # Get polarity score of the query
        print("The Query Polarity",query_polarity)
        # Compare products based on user query
        sentiment_analysis = compare_products(user_query)

        max_score = max(sentiment_rec["average_polarity"] for sentiment_rec in sentiment_analysis.values())

        if sentiment_analysis:
            # Display sentiment analysis results
            st.write("Sentiment Analysis Results:")
            best_product_data = None
            if query_polarity > 0:  # Positive sentiment query
                max_positive_score = -float('inf')
                for product, sentiment in sentiment_analysis.items():
                    score = sentiment["average_polarity"]
                    subjectivity = sentiment["average_subjectivity"]
                    Final_Sentiment = sentiment["overall_sentiment"]
                    Positive_Counts = sentiment["Positive_Count"]
                    Negative_Counts = sentiment["Negative_Count"]
                    Neutral_Counts = sentiment["Neutral_Count"]

                    # Convert subjectivity to a more understandable format
                    subjectivity_label = interpret_subjectivity(subjectivity)

                    if score >= max_positive_score:
                        max_positive_score = score
                        best_product_data = {
                            "Product": product,
                            "Sentiment_Analysis": Final_Sentiment,
                            "Subjectivity": subjectivity_label,
                            "Score":score,
                            "Positive_Counts": Positive_Counts,
                            "Negative_Counts": Negative_Counts,
                            "Neutral_Counts": Neutral_Counts

                        }

            elif query_polarity < 0:  # Negative sentiment query
                max_negative_score = float('inf')
                for product, sentiment in sentiment_analysis.items():
                    score = sentiment["average_polarity"]
                    subjectivity = sentiment["average_subjectivity"]
                    Final_Sentiment = sentiment["overall_sentiment"]
                    Positive_Counts = sentiment["Positive_Count"]
                    Negative_Counts = sentiment["Negative_Count"]
                    Neutral_Counts = sentiment["Neutral_Count"]

                    # Convert subjectivity to a more understandable format
                    subjectivity_label = interpret_subjectivity(subjectivity)

                    if score < max_negative_score:
                        max_negative_score = score
                        best_product_data = {
                            "Product": product,
                            "Sentiment_Analysis": Final_Sentiment,
                            "Subjectivity": subjectivity_label,
                            "Score":score,
                            "Positive_Counts": Positive_Counts,
                            "Negative_Counts": Negative_Counts,
                            "Neutral_Counts": Neutral_Counts
                        }

            if best_product_data:
                sentiment_summary = pd.DataFrame([best_product_data])
                st.write(sentiment_summary)

            # Visualization (optional): Bar chart of average polarities
            product_names = list(sentiment_analysis.keys())
            scores = [sentiment['average_polarity'] for sentiment in sentiment_analysis.values()]

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(product_names, scores, color='skyblue')
            ax.set_xlabel('Products')
            ax.set_ylabel('Average Polarity')
            ax.set_title('Average Polarity of Products')

            # Set x-axis ticks and labels
            ax.set_xticks(range(len(product_names)))
            ax.set_xticklabels(product_names, rotation=45, ha='right')

            st.pyplot(fig)

        else:
            st.write("No sentiment analysis results found.")

if __name__ == "__main__":
    main()