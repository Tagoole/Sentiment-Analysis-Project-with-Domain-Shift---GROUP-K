import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import re
import string

train_df = pd.read_csv('data/processed/processed_training_dataset.csv')
val_df = pd.read_csv('data/processed/processed_validation_datset.csv')
test_df = pd.read_csv('data/processed/student_test_dataset.csv')

SLANG_MAP = {
    r'\bu\b': 'you', r'\bpls\b': 'please', r'\bplz\b': 'please',
    r'\btmrw\b': 'tomorrow', r'\bwat\b': 'what', r'\bwud\b': 'would',
    r'\bcuz\b': 'because', r'\bcoz\b': 'because', r'\bgr8\b': 'great',
    r'\bbtw\b': 'by the way', r'\bidk\b': 'i do not know',
    r'\bomg\b': 'oh my god', r'\blol\b': 'laughing', r'\bthx\b': 'thanks',
    r'\bsry\b': 'sorry', r'\bwanna\b': 'want to', r'\bgonna\b': 'going to'
}

def clean(text):
    if not isinstance(text, str): return ''
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+|@\w+', '', text)
    for p, r in SLANG_MAP.items():
        text = re.sub(p, r, text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    return text.strip()

train_df['c'] = train_df['text'].apply(clean)
val_df['c'] = val_df['text'].apply(clean)
test_df['c'] = test_df['text'].apply(clean)

train_df = train_df[train_df['c'] != ''].reset_index(drop=True)
val_df = val_df[val_df['c'] != ''].reset_index(drop=True)
test_df = test_df[test_df['c'] != ''].reset_index(drop=True)

le = LabelEncoder()
y_train = le.fit_transform(train_df['sentiment'])
y_val = le.transform(val_df['sentiment'])
y_test = le.transform(test_df['sentiment'])

tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
X_train = tfidf.fit_transform(train_df['c'])
X_val = tfidf.transform(val_df['c'])
X_test = tfidf.transform(test_df['c'])

nb = MultinomialNB(alpha=0.1)
nb.fit(X_train, y_train)

print(f'NB Train Acc: {accuracy_score(y_train, nb.predict(X_train)):.4f}')
print(f'NB Val Acc:   {accuracy_score(y_val, nb.predict(X_val)):.4f}')
print(f'NB Test Acc:  {accuracy_score(y_test, nb.predict(X_test)):.4f}')
