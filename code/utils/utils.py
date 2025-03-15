import nltk
from nltk.corpus import stopwords
import re
import spacy 
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from nltk import ngrams
import random
import seaborn as sns



nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


############################################## Pre-processing functions ############################################

# -  Converts to lowercase, removes non-alphabetic characters, and eliminates stopwords.
def preprocess_text(text):
    text = text.lower()
    # This removes every thing but letters and blank spaces
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return tokens


# Another way to preprocess text to keep in a way the numbers in a more structured way (the high numbers will be replaced with <HIGH> and
# the smaller ones with <LOW>). We also going to add a custom stop word parameter in case we need it later.
def preprocess_text_nums(text, custom_stop_words=None):
    """
    Preprocesses text for NLP tasks by:
    - Converting to lowercase
    - Categorizing numbers as LOW/MEDIUM/HIGH and some more
    - Removing non-alphanumeric characters (except category placeholders)
    - Removing stop words
    
    Parameters:
    text (str): Input text to preprocess
    custom_stop_words (set, optional): Additional stop words to remove
    
    Returns:
    list: List of preprocessed tokens
    """
    if text is None or not isinstance(text, str):
        return ""
        
    text = text.lower()
    
    # Replace numbers with categories
    def replace_numbers(match):
        try:
            # Remove commas from number strings (American thousands separator)
            num_str = match.group().replace(',', '')
            num = float(num_str)
            
            # More granular categories for low values
            if num > 1000:
                return " HIGHNUMBER "
            elif 100 < num <= 1000:
                return " MEDIUMNUMBER "
            elif 50 < num <= 100:
                return " LOWNUMBER "
            elif 10 < num <= 50:
                return " VERYLOWNUMBER "
            elif 1 < num <= 10:
                return " EXTREMELYLOWNUMBER "
            else:  # num <= 1
                return " MINIMALNUMBER "
        except ValueError:
            return " NUM "

    # Match American format numbers: with optional commas as thousand separators and optional decimal point
    text = re.sub(r'\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?', replace_numbers, text)
    
    # Keep only lowercase and uppercase letters
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Handle multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize
    tokens = text.split()
    
    # Combine default and custom stop words if provided
    if custom_stop_words:
        if isinstance(custom_stop_words, set):
            all_stop_words = stop_words.union(custom_stop_words)
        else:
            all_stop_words = stop_words.union(set(custom_stop_words))
    else:
        all_stop_words = stop_words
    
    # Remove stop words
    tokens = [word for word in tokens if word not in all_stop_words]
    
    return tokens

# Now we do a function that tokenizes by n-grams:
def tokenize_ngrams(text, n=2):
    # First, preprocess the text using the existing function
    tokens = preprocess_text(text)
    
    # Generate n-grams
    n_grams = list(ngrams(tokens, n))
    
    # Join the n-grams into strings
    n_gram_tokens = [' '.join(gram) for gram in n_grams]
    
    return n_gram_tokens

############################################# NER ###########################################

# Function to extract named entities
def extract_entities(text, model):
    doc = model(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

# Function to get most frequent entities of specific types
def extract_most_frequent_entities(entities, target_labels):
    """Filters and counts most frequent entities of given types."""
    filtered_entities = [ent[0] for ent in entities if ent[1] in target_labels]
    return Counter(filtered_entities).most_common(10)

# For plotting the results:
def plot_entity_frequencies(entity_counts, model_name):
    """Plots the most frequent entities using a bar chart."""
    entities, counts = zip(*entity_counts)  # Unpack tuples into lists

    plt.figure(figsize=(10, 5))
    plt.barh(entities[::-1], counts[::-1], color='skyblue')  # Reverse for better display
    plt.xlabel("Occurrences")
    plt.ylabel("Entities")
    plt.title(f"Most Frequent Named Entities ({model_name})")
    plt.show()



############################################# TF-IDF ###########################################

# Basic TF-IDF function:
# Function to get top TF-IDF terms for each sentiment
def basic_top_tfidf_terms(df, column='Notices', label_column='Y', n_terms=20):
    # Create a corpus for each sentiment category
    sentiment_categories = df[label_column].unique()
    corpus_by_sentiment = {}
    
    for sentiment in sentiment_categories:
        subset = df[df[label_column] == sentiment]
        # Preprocess each document and join tokens back to strings for the vectorizer
        corpus_by_sentiment[sentiment] = [
            ' '.join(preprocess_text_nums(text)) 
            for text in subset[column]
        ]
    
    # Flatten all documents for TF-IDF calculation
    all_docs = []
    for sentiment in sentiment_categories:
        all_docs.extend(corpus_by_sentiment[sentiment])
    
    # Create and fit TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=10000,  # Limit to top 10,000 features to manage memory
        min_df=2,            # Ignore terms that appear in fewer than 2 documents
        max_df=0.95,         # Ignore terms that appear in more than 95% of documents
        ngram_range=(1, 1)   # Only use unigrams
    )
    
    tfidf_matrix = vectorizer.fit_transform(all_docs)
    feature_names = vectorizer.get_feature_names_out()
    
    # Get top terms for each sentiment category
    top_terms = {}
    start_idx = 0
    
    for sentiment in sentiment_categories:
        docs_count = len(corpus_by_sentiment[sentiment])
        if docs_count == 0:
            continue
            
        # Get the TF-IDF scores for this sentiment's documents
        end_idx = start_idx + docs_count
        sentiment_tfidf = tfidf_matrix[start_idx:end_idx]
        start_idx = end_idx
        
        # Calculate average TF-IDF score for each term across all documents of this sentiment
        avg_tfidf = np.asarray(sentiment_tfidf.mean(axis=0)).flatten()
        
        # Get indices of top terms
        top_indices = avg_tfidf.argsort()[-n_terms:][::-1]
        
        # Store top terms and their scores
        top_terms[sentiment] = [(feature_names[i], avg_tfidf[i]) for i in top_indices]
    
    return top_terms

# Function to create merged documents for each sentiment class
def create_merged_docs(df, column='Notices', label_column='Y', chunk_size=10):
    """
    Creates merged documents for each sentiment class.
    
    Args:
        df: DataFrame containing the text data
        column: Column name containing the text
        label_column: Column name containing the sentiment labels
        chunk_size: Number of documents to merge into one chunk
    
    Returns:
        Dictionary with sentiment classes as keys and lists of merged documents as values
    """
    sentiment_categories = df[label_column].unique()
    merged_docs_by_sentiment = {}
    
    for sentiment in sentiment_categories:
        subset = df[df[label_column] == sentiment]
        texts = subset[column].tolist()
        
        # Shuffle the texts to get a random mix
        random.seed(42)  # For reproducibility
        random.shuffle(texts)
        
        # Create chunks of texts
        chunks = []
        for i in range(0, len(texts), chunk_size):
            chunk = ' '.join(texts[i:i+chunk_size])
            chunks.append(chunk)
        
        merged_docs_by_sentiment[sentiment] = chunks
    
    return merged_docs_by_sentiment

# Function to create a single merged document for each sentiment class
def create_single_merged_docs(df, column='Notices', label_column='Y'):
    """
    Creates a single merged document for each sentiment class.
    
    Args:
        df: DataFrame containing the text data
        column: Column name containing the text
        label_column: Column name containing the sentiment labels
    
    Returns:
        Dictionary with sentiment classes as keys and merged documents as values
    """
    sentiment_categories = df[label_column].unique()
    merged_docs_by_sentiment = {}
    
    for sentiment in sentiment_categories:
        subset = df[df[label_column] == sentiment]
        merged_doc = ' '.join(subset[column].tolist())
        merged_docs_by_sentiment[sentiment] = [merged_doc]  # Wrap in list for consistency
    
    return merged_docs_by_sentiment

# Function to get top TF-IDF terms for each sentiment
def get_top_tfidf_terms(corpus_by_sentiment, n_terms=20):
    """
    Gets the top TF-IDF terms for each sentiment.
    
    Args:
        corpus_by_sentiment: Dictionary with sentiment classes as keys and lists of preprocessed documents as values
        n_terms: Number of top terms to return
    
    Returns:
        Dictionary with sentiment classes as keys and lists of (term, score) tuples as values
    """
    # Flatten all documents for TF-IDF calculation
    all_docs = []
    for sentiment, docs in corpus_by_sentiment.items():
        all_docs.extend(docs)
    
    # Create and fit TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=10000,  # Limit to top 10,000 features to manage memory
        min_df=2,            # Ignore terms that appear in fewer than 2 documents
        max_df=0.95,         # Ignore terms that appear in more than 95% of documents
        ngram_range=(1, 1)   # Only use unigrams
    )
    
    tfidf_matrix = vectorizer.fit_transform(all_docs)
    feature_names = vectorizer.get_feature_names_out()
    
    # Get top terms for each sentiment category
    top_terms = {}
    start_idx = 0
    
    for sentiment, docs in corpus_by_sentiment.items():
        docs_count = len(docs)
        if docs_count == 0:
            continue
            
        # Get the TF-IDF scores for this sentiment's documents
        end_idx = start_idx + docs_count
        sentiment_tfidf = tfidf_matrix[start_idx:end_idx]
        start_idx = end_idx
        
        # Calculate average TF-IDF score for each term across all documents of this sentiment
        avg_tfidf = np.asarray(sentiment_tfidf.mean(axis=0)).flatten()
        
        # Get indices of top terms
        top_indices = avg_tfidf.argsort()[-n_terms:][::-1]
        
        # Store top terms and their scores
        top_terms[sentiment] = [(feature_names[i], avg_tfidf[i]) for i in top_indices]
    
    return top_terms

# Main code to run the analysis
def run_tfidf_analysis(df, column='Notices', label_column='Y', n_terms=20):
    # Preprocess all texts
    df['processed_text'] = df[column].apply(lambda x: ' '.join(preprocess_text_nums(x)))
    
    # Approach 1: TF-IDF on merged chunks (10 documents per chunk)
    merged_docs = create_merged_docs(df, column='processed_text', label_column=label_column, chunk_size=10)
    top_terms_chunks = get_top_tfidf_terms(merged_docs, n_terms=n_terms)
    
    # Approach 2: TF-IDF on single merged document per class
    single_merged_docs = create_single_merged_docs(df, column='processed_text', label_column=label_column)
    top_terms_single = get_top_tfidf_terms(single_merged_docs, n_terms=n_terms)
    
    return top_terms_chunks, top_terms_single

# Visualize the results
def visualize_results(top_terms_chunks, top_terms_single):
    # Visualize top terms from chunks
    plt.figure(figsize=(15, 4 * len(top_terms_chunks)))
    plt.suptitle("Top TF-IDF Terms Using Merged Chunks", fontsize=16)
    
    for i, (sentiment, terms) in enumerate(top_terms_chunks.items()):
        plt.subplot(len(top_terms_chunks), 1, i+1)
        
        terms_df = pd.DataFrame(terms, columns=['term', 'tfidf'])
        sns.barplot(x='tfidf', y='term', data=terms_df, palette='viridis')
        
        plt.title(f"Sentiment: '{sentiment}'")
        plt.tight_layout()
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()
    
    # Visualize top terms from single merged document
    plt.figure(figsize=(15, 4 * len(top_terms_single)))
    plt.suptitle("Top TF-IDF Terms Using Single Merged Document Per Class", fontsize=16)
    
    for i, (sentiment, terms) in enumerate(top_terms_single.items()):
        plt.subplot(len(top_terms_single), 1, i+1)
        
        terms_df = pd.DataFrame(terms, columns=['term', 'tfidf'])
        sns.barplot(x='tfidf', y='term', data=terms_df, palette='viridis')
        
        plt.title(f"Sentiment: '{sentiment}'")
        plt.tight_layout()
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()