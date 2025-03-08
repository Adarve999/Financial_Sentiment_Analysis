import nltk
from nltk.corpus import stopwords
import re
import spacy 
from collections import Counter
import matplotlib.pyplot as plt

# nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


# -  Converts to lowercase, removes non-alphabetic characters, and eliminates stopwords.
def preprocess_text(text):
    text = text.lower()
    # Elimina todo lo que no sean letras o espacios
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
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
