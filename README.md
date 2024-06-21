# Product_Sentiment_Analysis

# Overview
This Streamlit app performs sentiment analysis on product reviews stored in CSV files. It uses TextBlob for sentiment analysis, spaCy for tokenization, and scikit-learn for TF-IDF vectorization and cosine similarity calculation.

# Features
Sentiment Analysis: Analyze sentiment (positive, negative, neutral) and subjectivity of product reviews.
Comparison: Compare multiple products based on user queries.
Visualization: Display the average polarity of products using a bar chart.

# Usage
Run the Streamlit app
streamlit run app.py

Open a web browser and go to http://localhost:8501 to view the app.

Enter a query in the text box to analyze the sentiment of products related to the query.

View the sentiment analysis results including the best matching product based on sentiment and subjectivity.

Explore the bar chart to see the average polarity of products.

# Libraries Used
The app utilizes several Python libraries:

Streamlit: For building and serving the web application.

Pandas: For data manipulation and analysis.

TextBlob: For natural language processing tasks such as sentiment analysis.

Matplotlib: For data visualization, specifically to create the bar chart.

spaCy: For advanced natural language processing tasks such as tokenization.

scikit-learn: For TF-IDF vectorization and cosine similarity calculations.


