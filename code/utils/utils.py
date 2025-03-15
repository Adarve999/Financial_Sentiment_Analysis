import nltk
from nltk.corpus import stopwords
import re
import spacy 
from collections import Counter
import matplotlib.pyplot as plt

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


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
