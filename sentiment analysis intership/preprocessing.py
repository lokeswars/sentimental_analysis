import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import joblib

nltk.download('stopwords')
nltk.download('punkt')

def clean_text(text):
    # Remove emails
    text = ' '.join([i for i in text.split() if '@' not in i])
    # Remove web addresses
    text = re.sub('http[s]?://\S+', '', text)
    # Filter to allow only alphabets and single quotes
    text = re.sub(r'[^a-zA-Z\']', ' ', text)
    # Remove Unicode characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    # Convert to lowercase to maintain consistency
    text = text.lower()
    # Remove double spaces
    text = re.sub('\s+', ' ', text)
    return text

def stop_free(text):
    tokens = word_tokenize(text)
    df = pd.DataFrame(tokens, columns=['cleaned_tweet'])
    df['cleaned_tweet'] = df['cleaned_tweet'].apply(lambda x: ' '.join([word for word in x.split() if word not in stopwords.words('english')]))
    cleaned_text = " ".join(df['cleaned_tweet'].tolist())  # Combine tokens into cleaned text
    return cleaned_text

from sklearn.feature_extraction.text import CountVectorizer

def vec(df):
    vect=joblib.load('vect.pkl')
    
    # Fit and transform your data
    vectorized = vect.transform(df['cleaned_tweet'])
    return vectorized






'''
# Load your model
model = joblib.load('model.pkl')

text = "#india @kingjan follow me to welcome"
ct = clean_text(text)
stop_free_text = stop_free(ct)

# Now you can vectorize your text
vectorized = vec(pd.DataFrame([stop_free_text], columns=['cleaned_tweet']))

rf = model.predict(vectorized)

'''