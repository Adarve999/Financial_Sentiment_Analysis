import nltk
from nltk.corpus import stopwords
import re

# nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


# - Pasa a minúsculas, elimina caracteres que no sean letras y remueve stopwords.
def preprocess_text(text):
    text = text.lower()
    # Elimina todo lo que no sean letras o espacios
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return tokens