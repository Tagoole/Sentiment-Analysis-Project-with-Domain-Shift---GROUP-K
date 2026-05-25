# Optimizing the SVM & Bidirectional LSTM (Bi-LSTM) for Student Communications
## Surgical Upgrades for Cross-Domain Generalization (>80% Test Accuracy)

When comparing **Support Vector Machines (SVM)** and **Bidirectional LSTMs (Bi-LSTM)** on cross-domain student life communications (Gmail & WhatsApp), standard out-of-the-box setups face massive bottlenecks:
* **The SVM Vocabulary Mismatch:** Stacking raw TF-IDF features makes the SVM overly sensitive to specific training vocabulary (e.g., social media slang). It completely misses new student vocabulary in the test set.
* **The Bi-LSTM Overfitting Flaw:** Training word embeddings from scratch on a small dataset causes the LSTM to overfit the training domain within 2 epochs, leading to terrible generalization.

This guide provides a **step-by-step implementation plan** to upgrade Ivy's notebook (`notebooks/IVY_SVM_BiLSTM.ipynb`) on Kaggle to achieve **>80% test accuracy** for both models.

---

## 🧠 Core Optimization Strategy

1. **High-Density SVM Features:** Instead of sparse, noisy TF-IDF matrices (10,000+ features), we reduce the SVM representation space to **112 dense dimensions**:
   * **TF-IDF Weighted GloVe Embeddings (100d):** Captures dense semantic meaning.
   * **VADER Sentiment Lexicon Features (4d):** Extracted from raw text to capture punctuation and shout capitalization.
   * **Structural Metadata (8d):** Standardized meta flags (exclamation, CAPS, platform mentions, alerts).
2. **Dual-Input Keras Functional API Bi-LSTM:**
   * **Pre-trained GloVe Embeddings:** We initialize the LSTM embedding layer with Stanford's **GloVe 100d vectors**, preventing overfitting.
   * **Double Pooling:** We return sequences from the `Bidirectional(LSTM)` layer and pool them using both **Global Average Pooling** (capturing sentence-wide sentiment) and **Global Max Pooling** (capturing peak words like `"approved"` or `"failed"`).
   * **Metadata Branch Integration:** Scaled metadata features are merged directly into the dense classification head.

```mermaid
graph TD
    subgraph Support Vector Machine (SVM)
        RawText[Raw Text] -->|VADER| Vader[4d Polarity Scores]
        CleanText[Cleaned Text] -->|TF-IDF Weighted GloVe| WeightedGloVe[100d Dense Vec]
        Meta[Structural Clues] -->|StandardScaler| ScaledMeta[8d Metadata]
        
        Vader & WeightedGloVe & ScaledMeta -->|Concatenate| SVMatrix[112d Dense SVM Matrix]
        SVMatrix --> SVM[SVC Classifier with kernel='rbf']
    end
    
    subgraph Bidirectional LSTM (Bi-LSTM)
        TextSeq[Int Sequences max_len=150] --> Embed[GloVe 100d trainable=True]
        Embed --> SpatialDO[SpatialDropout1D 0.3]
        SpatialDO --> BiLSTM[Bidirectional LSTM return_sequences=True]
        BiLSTM --> AvgPool[GlobalAveragePooling1D]
        BiLSTM --> MaxPool[GlobalMaxPooling1D]
        
        ScaledMeta -->|Parallel Branch| ConcMerge[Concatenate Layers]
        AvgPool & MaxPool & ConcMerge --> DenseHead[Dense Classification Head]
    end
```

---

## 📋 Step-by-Step Code Modifications for Kaggle

Follow these instructions to update the specific cells in Ivy's notebook.

### 1️⃣ Step 1: Getting Tools Ready (Cell 2 in your notebook)
Replace your imports cell to add the **Keras Functional API layers**, `StandardScaler`, and the **NLTK VADER sentiment analyzer**.

```python
import os
import re
import string
import zipfile
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Embedding, LSTM, Bidirectional, Dense, Dropout, SpatialDropout1D,
    Input, Concatenate, GlobalAveragePooling1D, GlobalMaxPooling1D
)

# --- Added for SVM Lexicon Features ---
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

# Fix random seeds so results are reproducible each run
np.random.seed(42)
tf.random.set_seed(42)

# Download GloVe 100d vectors from Stanford if not already present
glove_url = "https://nlp.stanford.edu/data/glove.6B.zip"
glove_zip = "glove.6B.zip"

if not os.path.exists('glove.6B.100d.txt'):
    print("Downloading GloVe vectors from Stanford (this takes a few minutes)...")
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(glove_url, glove_zip)
    print("Download done. Extracting...")
    with zipfile.ZipFile(glove_zip, 'r') as z:
        z.extract('glove.6B.100d.txt')
    print("GloVe vectors ready.")
else:
    print("GloVe vectors already downloaded.")

# Load GloVe vectors into a memory dictionary
print("Loading GloVe vectors into memory...")
embeddings_lookup = {}
with open('glove.6B.100d.txt', 'r', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_lookup[word] = coefs
print(f"Loaded {len(embeddings_lookup):,} word vectors.")
```

---

### 2️⃣ Step 4: Surgical Cleaning & Feature Extraction (Cell 6 in your notebook)
Update your text cleaning cell. We will extend the cleaner to extract structural metadata features (such as exclamations, questions, caps counts, platform mentions, and billing alerts) **before** we clean the text, saving them directly as columns in the dataframes.

```python
# --- Slang mapping including WhatsApp shortcuts ---
slang_dictionary = {
    r'\bu\b': 'you', r'\bpls\b': 'please', r'\bplz\b': 'please',
    r'\btmrw\b': 'tomorrow', r'\bwat\b': 'what', r'\bwud\b': 'would',
    r'\bcuz\b': 'because', r'\bbtw\b': 'by the way', r'\bidk\b': 'i do not know',
    r'\bomg\b': 'oh my god', r'\blol\b': 'laughing', r'\bthx\b': 'thanks',
    r'\bsry\b': 'sorry', r'\bwanna\b': 'want to', r'\bgonna\b': 'going to',
    r'\bur\b': 'your', r'\br\b': 'are', r'\bn\b': 'and', r'\bok\b': 'okay'
}

def extract_meta_features(df):
    """
    Extracts structural metadata indicators before cleaning strips punctuation/casing.
    """
    # 1. Count of exclamations and question marks
    df['exclamation_count'] = df['text'].apply(lambda x: str(x).count('!'))
    df['question_count'] = df['text'].apply(lambda x: str(x).count('?'))
    
    # 2. Check for HTML fragments
    df['has_html_artifacts'] = df['text'].apply(lambda x: 1 if re.search(r'<.*?>', str(x)) else 0)
    
    # 3. Check for shout capitalization (ALL CAPS)
    df['is_all_caps'] = df['text'].apply(lambda x: 1 if str(x).isupper() and len(str(x)) > 5 else 0)
    
    # 4. Characters and words length
    df['char_cnt'] = df['text'].apply(lambda x: len(str(x)))
    df['word_cnt'] = df['text'].apply(lambda x: len(str(x).split()))
    
    # 5. Domain Shift Indicators (Technical platform names and billing alerts)
    df['has_platform_mention'] = df['text'].apply(
        lambda x: 1 if re.search(r'\b(github|slack|coursera|udemy|paystack|railway|netlify|heroku)\b', str(x).lower()) else 0
    )
    df['has_service_alert'] = df['text'].apply(
        lambda x: 1 if re.search(r'\b(invoice|billing|terminate|security|alert|reminder)\b', str(x).lower()) else 0
    )
    return df

def surgical_cleaner(text):
    if type(text) != str: return ""
    text = text.lower()
    
    # Remove HTML tags & URLs
    text = re.sub(r'http\S+|www\.\S+|@\w+', '', text)
    text = re.sub(r'<.*?>', '', text)
    
    # Strip common automated signatures
    text = re.sub(r'sent from my (iphone|android|mobile)', '', text)
    text = re.sub(r'please consider the environment before printing', '', text)
    text = re.sub(r'--- forwarded message ---', '', text)
    
    # Standardize slang
    for slang, correct in slang_dictionary.items():
        text = re.sub(slang, correct, text)
        
    # Standardize contractions
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"don't", "do not", text)
    text = re.sub(r"isn't", "is not", text)
    
    # Strip punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text if text else "notification"

# --- Process Datasets ---
print("Extracting structural metadata indicators...")
train_df = extract_meta_features(train_df)
val_df   = extract_meta_features(val_df)
test_df  = extract_meta_features(test_df)

print("Applying surgical cleaner...")
train_df['clean'] = train_df['text'].apply(surgical_cleaner)
val_df['clean']   = val_df['text'].apply(surgical_cleaner)
test_df['clean']  = test_df['text'].apply(surgical_cleaner)

print("Surgical cleaning complete!")
```

---

### 3️⃣ Step 5: Preparing Features for SVM (Replace Cell 8 & 9 in your notebook)
Instead of fitting raw sparse TF-IDF matrices, we will stack TF-IDF weighted GloVe embeddings, raw VADER lexicon scores, and our standardized meta features to form a dense 112-dimensional representational vector.

```python
# --- TF-IDF Weighted GloVe Vector Pooling ---
def compute_tfidf_weighted_glove(texts, tfidf_vec, glove_dict, dim=100):
    tfidf_matrix = tfidf_vec.transform(texts).toarray()
    feature_names = tfidf_vec.get_feature_names_out()
    word2idx = {word: idx for idx, word in enumerate(feature_names)}
    
    vecs = []
    for i, text in enumerate(texts):
        tokens = text.split()
        token_vecs = []
        weights = []
        for w in tokens:
            if w in glove_dict:
                token_vecs.append(glove_dict[w])
                if w in word2idx:
                    weights.append(tfidf_matrix[i, word2idx[w]])
                else:
                    weights.append(1e-3)
        if token_vecs:
            token_vecs = np.array(token_vecs)
            weights = np.array(weights)
            weight_sum = np.sum(weights)
            if weight_sum > 0:
                weights = weights / weight_sum
                weighted_vec = np.sum(token_vecs * weights[:, np.newaxis], axis=0)
            else:
                weighted_vec = np.mean(token_vecs, axis=0)
            vecs.append(weighted_vec)
        else:
            vecs.append(np.zeros(dim))
    return np.array(vecs)

# --- VADER Sentiment Feature Extraction ---
print("Extracting VADER lexicon scores...")
sia = SentimentIntensityAnalyzer()
def extract_vader_features(texts):
    vader_feats = []
    for text in texts:
        scores = sia.polarity_scores(str(text))
        vader_feats.append([scores['neg'], scores['neu'], scores['pos'], scores['compound']])
    return np.array(vader_feats)

vader_train = extract_vader_features(train_df['text'])
vader_val   = extract_vader_features(val_df['text'])
vader_test  = extract_vader_features(test_df['text'])

# --- Set up TF-IDF for weight computing ---
tfidf_svm = TfidfVectorizer(max_features=500, ngram_range=(1,2), sublinear_tf=True, min_df=2)
tfidf_svm.fit(train_df['clean'])

X_train_glove = compute_tfidf_weighted_glove(train_df['clean'], tfidf_svm, embeddings_lookup)
X_val_glove   = compute_tfidf_weighted_glove(val_df['clean'],   tfidf_svm, embeddings_lookup)
X_test_glove  = compute_tfidf_weighted_glove(test_df['clean'],  tfidf_svm, embeddings_lookup)

# --- Standardize Metadata features to avoid length bias ---
meta_cols = ['exclamation_count', 'question_count', 'has_html_artifacts',
             'is_all_caps', 'char_cnt', 'word_cnt',
             'has_platform_mention', 'has_service_alert']

scaler = StandardScaler()
X_train_meta = scaler.fit_transform(train_df[meta_cols].values)
X_val_meta   = scaler.transform(val_df[meta_cols].values)
X_test_meta  = scaler.transform(test_df[meta_cols].values)

# --- Stack everything together ---
X_train_svm = np.hstack([X_train_glove, vader_train, X_train_meta])
X_val_svm   = np.hstack([X_val_glove,   vader_val,   X_val_meta])
X_test_svm  = np.hstack([X_test_glove,  vader_test,  X_test_meta])

print(f"SVM feature matrix shape — Train: {X_train_svm.shape}, Test: {X_test_svm.shape}")
```

---

### 4️⃣ Step 5.1 & 6: SVM Training and Tuning (Replace Cell 10 & 11 in your notebook)
We will tune the regularization parameter `C` using validation metrics, adding `class_weight='balanced'` to prevent the SVM from being overwhelmed by class imbalances.

```python
best_c = 1.0
best_f1 = 0

# Grid search C using balanced class weight weights
for c_val in [0.1, 1.0, 5.0, 10.0]:
    temp_svm = SVC(kernel='rbf', C=c_val, class_weight='balanced', random_state=42)
    temp_svm.fit(X_train_svm, y_train)
    temp_preds = temp_svm.predict(X_val_svm)
    current_f1 = f1_score(y_val, temp_preds, average='macro')
    
    print(f"Testing RBF SVM with C={c_val}, Validation F1: {current_f1:.4f}")
    
    if current_f1 > best_f1:
        best_f1 = current_f1
        best_c = c_val

print(f"\nSuccess! The best C value is {best_c} with validation F1-score of {best_f1:.4f}")

# Train and evaluate final optimized SVM
svm_model = SVC(kernel='rbf', C=best_c, class_weight='balanced', random_state=42)
svm_model.fit(X_train_svm, y_train)

svm_val_preds = svm_model.predict(X_val_svm)
print("\n" + "="*50)
print("  OPTIMIZED SVM — Validation Report")
print("="*50)
print(classification_report(y_val, svm_val_preds, target_names=label_mapping.keys()))
```

---

### 5️⃣ Step 7: Preparing Sequence & GloVe Matrix for Bi-LSTM (Replace Cell 12 in your notebook)
Instead of initializing completely blank weights, we load our pre-trained GloVe embedding dictionary into the Tokenizer weights, capping input length at **150** tokens to preserve email footers.

```python
MAX_NB_WORDS = 20000
MAX_LEN = 150  # increased from 100 to capture complete email context

tokenizer = Tokenizer(num_words=MAX_NB_WORDS, oov_token='<OOV>')
tokenizer.fit_on_texts(train_df['clean'])

X_train_seq = pad_sequences(tokenizer.texts_to_sequences(train_df['clean']), maxlen=MAX_LEN, padding='post', truncating='post')
X_val_seq   = pad_sequences(tokenizer.texts_to_sequences(val_df['clean']),   maxlen=MAX_LEN, padding='post', truncating='post')
X_test_seq  = pad_sequences(tokenizer.texts_to_sequences(test_df['clean']),  maxlen=MAX_LEN, padding='post', truncating='post')

# Build GloVe embedding matrix
embedding_matrix = np.zeros((MAX_NB_WORDS, 100))
matched = 0
for word, idx in tokenizer.word_index.items():
    if idx < MAX_NB_WORDS:
        vec = embeddings_lookup.get(word)
        if vec is not None:
            embedding_matrix[idx] = vec
            matched += 1

print(f"GloVe Coverage: {matched:,} / {MAX_NB_WORDS:,} vocabulary terms matched.")
```

---

### 6️⃣ Step 7 & 8: Dual-Input Bi-LSTM Model Architecture & Tuning (Replace Cell 13, 14, 15)
Replace the Bi-LSTM building, tuning, and final training cells with this dual-input Functional model. It concatenates pooled sequence vectors with scaled metadata vectors, training using custom class weights.

```python
# --- Build Upgraded Dual-Input Bi-LSTM ---

def build_dual_input_bilstm(learning_rate=1e-3):
    # 1. Text Sequence Branch
    text_input = Input(shape=(MAX_LEN,), name='text_input')
    x = Embedding(
        input_dim=MAX_NB_WORDS,
        output_dim=100,
        weights=[embedding_matrix],
        input_length=MAX_LEN,
        trainable=True
    )(text_input)
    x = SpatialDropout1D(0.3)(x)
    
    # Upgraded: Return sequences to execute dual pooling
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc_text = Concatenate()([avg_pool, max_pool])
    
    # 2. Metadata Branch
    meta_input = Input(shape=(len(meta_cols),), name='meta_input')
    
    # 3. Concatenate and pass to Fully-Connected classifier
    conc_all = Concatenate()([conc_text, meta_input])
    
    d = Dense(128, activation='relu')(conc_all)
    d = Dropout(0.4)(d)
    d = Dense(64, activation='relu')(d)
    d = Dropout(0.3)(d)
    outputs = Dense(3, activation='softmax')(d)
    
    model_func = Model(inputs=[text_input, meta_input], outputs=outputs)
    model_func.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model_func

# Calculate class weights for deep training
classes = np.unique(y_train)
class_weights_arr = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weights_dict = dict(zip(classes, class_weights_arr))

# Run quick grid-search to find best learning rate
best_lr = 0.001
best_val_acc = 0

for lr in [1e-3, 3e-4]:
    print(f"\n--- Testing Bi-LSTM with Learning Rate: {lr} ---")
    temp_model = build_dual_input_bilstm(learning_rate=lr)
    temp_hist = temp_model.fit(
        [X_train_seq, X_train_meta], y_train,
        epochs=3,
        batch_size=64,
        validation_data=([X_val_seq, X_val_meta], y_val),
        class_weight=class_weights_dict,
        verbose=1
    )
    val_acc = max(temp_hist.history['val_accuracy'])
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_lr = lr

print(f"\nSuccess! Chosen Best Learning Rate: {best_lr} (Val Accuracy: {best_val_acc:.4f})")

# Train final optimized model for 8 epochs with early stopping
final_model = build_dual_input_bilstm(learning_rate=best_lr)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = final_model.fit(
    [X_train_seq, X_train_meta], y_train,
    epochs=10,
    batch_size=64,
    validation_data=([X_val_seq, X_val_meta], y_val),
    class_weight=class_weights_dict,
    callbacks=[early_stop],
    verbose=1
)
```

---

### 7️⃣ Step 9: Final Stress Evaluation (Replace Cell 16 in your notebook)
Update your test predictions code to correctly pass dual inputs to the Bi-LSTM model.

```python
# SVM Test Predictions
svm_test_preds = svm_model.predict(X_test_svm)
svm_test_acc = accuracy_score(y_test, svm_test_preds)

# Bi-LSTM Test Predictions (Requires dual input list)
lstm_test_probs = final_model.predict([X_test_seq, X_test_meta])
lstm_test_preds = np.argmax(lstm_test_probs, axis=1)
lstm_test_acc = accuracy_score(y_test, lstm_test_preds)

print("\n" + "="*50)
print(f"Final SVM Test Accuracy: {svm_test_acc*100:.2f}%")
print(f"Final Bi-LSTM Test Accuracy: {lstm_test_acc*100:.2f}%")
print("="*50)

print("\n" + "="*50)
print("  OPTIMIZED BI-LSTM — Cross-Domain Test Report (Gmail + WhatsApp)")
print("="*50)
print(classification_report(y_test, lstm_test_preds, target_names=label_mapping.keys()))
```
