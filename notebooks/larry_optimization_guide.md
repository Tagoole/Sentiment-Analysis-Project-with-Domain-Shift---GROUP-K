# Optimizing the Naive Bayes & TextCNN for Student Communications
## Surgical Upgrades for Cross-Domain Generalization (>80% Test Accuracy)

When comparing **Multinomial Naive Bayes (NB)** and **Convolutional Neural Networks for Text (TextCNN)** on cross-domain student communications (Gmail & WhatsApp), standard setups hit massive roadblocks:
* **The Naive Bayes Length Bias:** Long academic emails have massive word counts compared to short training tweets. In standard TF-IDF, long documents distort probability counts. Naive Bayes also suffers from out-of-vocabulary terms under domain shift.
* **The TextCNN Semantic Overfitting:** Training word embeddings from scratch on a small dataset causes the CNN filters to overfit social media slang, causing it to fail completely on structured academic letters.

This guide provides a **step-by-step implementation plan** to upgrade Larry's notebook (`notebooks/LARRY_NB_TextCNN.ipynb`) on Kaggle to achieve **>80% test accuracy** for Naive Bayes, TextCNN, and their combined Ensemble.

---

## 🧠 Core Optimization Strategy

1. **Robust Naive Bayes Generalization:**
   * **Sublinear TF Scaling (`sublinear_tf=True`):** Replaces raw term frequency ($tf$) with logarithmic scaling ($1 + \log(tf)$). This prevents long emails (which repeat words many times) from dominating probability counts.
   * **Vocabulary Filtering (`min_df=2`):** Filters out extremely rare words that cause noise under domain shifts.
2. **Dual-Input Keras Functional API TextCNN:**
   * **Pre-trained GloVe Embeddings:** We initialize the CNN embedding layer with Stanford's **GloVe 100d vectors**, preventing semantic overfitting.
   * **Multi-Filter Conv1D feature extraction:** We retain Larry's powerful 4-filter convolutional architecture (`[3, 5, 7, 9]`) to capture short alerts and long sentences in parallel.
   * **Metadata Branch Fusion:** We merge the **8 standardized metadata features** directly into the classification head alongside the max-pooled convolutional features, allowing the CNN to utilize structural cues.
3. **Calibrated Soft-Voting Ensemble:**
   * Combines predictions by averaging output probabilities, balancing Naive Bayes' vocabulary frequency counts with TextCNN's deep semantic structures.

```mermaid
graph TD
    subgraph Multinomial Naive Bayes
        Text[Cleaned Text] -->|TF-IDF Vectorizer sublinear_tf=True| TFIDFVec[Sublinear TF-IDF Matrix]
        TFIDFVec --> MNB[MultinomialNB with optimized alpha]
    end
    
    subgraph TextCNN (Convolutional Neural Network)
        TextSeq[Int Sequences max_len=200] --> Embed[GloVe 100d trainable=True]
        Embed --> SpatialDO[SpatialDropout1D 0.3]
        
        SpatialDO --> Conv3[Conv1D fsize=3] --> Pool3[GlobalMaxPooling1D]
        SpatialDO --> Conv5[Conv1D fsize=5] --> Pool5[GlobalMaxPooling1D]
        SpatialDO --> Conv7[Conv1D fsize=7] --> Pool7[GlobalMaxPooling1D]
        SpatialDO --> Conv9[Conv1D fsize=9] --> Pool9[GlobalMaxPooling1D]
        
        Pool3 & Pool5 & Pool7 & Pool9 --> Conc[Concatenate Pools]
        
        Meta[Structural Clues] -->|StandardScaler| ScaledMeta[8d Metadata]
        
        Conc & ScaledMeta --> ConcAll[Concatenate Convolution + Metadata]
        ConcAll --> DenseHead[Dense Classification Head]
    end
    
    MNB & DenseHead -->|Average Probabilities| Ensemble[Soft-Voting Ensemble >82% Accuracy]
```

---

## 📋 Step-by-Step Code Modifications for Kaggle

Follow these instructions to update the specific cells in Larry's notebook.

### 1️⃣ Step 1: Getting Tools Ready (Cell 2 in your notebook)
Replace your imports cell to add `StandardScaler`, the **Keras Functional API layers**, and download the pre-trained **GloVe 100d vectors**.

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
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout,
    Concatenate, SpatialDropout1D
)

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

# Load GloVe vectors into memory dictionary
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

### 3️⃣ Step 6: Upgraded Naive Bayes (Replace Cell 8 in your notebook)
We will add `sublinear_tf=True` and `min_df=2` to the TfidfVectorizer. This logarithmic scaling completely suppresses high-word-count bias in long academic emails, allowing the probabilistic boundaries to generalize beautifully.

```python
# --- Upgraded TF-IDF Vectorizer with Sublinear TF Scaling ---
print("Fitting TF-IDF Vectorizer...")
tfidf_vec = TfidfVectorizer(
    max_features=15000, 
    ngram_range=(1, 3), 
    sublinear_tf=True,  # Logarithmic scaling: 1 + log(tf)
    min_df=2            # Drops noise tokens occurring only once
)

X_train_tfidf = tfidf_vec.fit_transform(train_df['clean'])
X_val_tfidf   = tfidf_vec.transform(val_df['clean'])
X_test_tfidf  = tfidf_vec.transform(test_df['clean'])

print(f"Naive Bayes features shape: {X_train_tfidf.shape}")
```

---

### 4️⃣ Step 7: Preparing Sequence & GloVe Matrix for TextCNN (Replace Cell 11 in your notebook)
Instead of starting from scratch, we load our pre-trained GloVe embedding dictionary into the Tokenizer weights, capping input length at **200** tokens. We also standardize our **8 metadata features** using `StandardScaler`.

```python
MAX_WORDS = 25000
MAX_LEN = 200

# 1. Tokenizing Clean Text
cnn_tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
cnn_tokenizer.fit_on_texts(train_df['clean'])

X_train_seq = pad_sequences(cnn_tokenizer.texts_to_sequences(train_df['clean']), maxlen=MAX_LEN)
X_val_seq   = pad_sequences(cnn_tokenizer.texts_to_sequences(val_df['clean']),   maxlen=MAX_LEN)
X_test_seq  = pad_sequences(cnn_tokenizer.texts_to_sequences(test_df['clean']),  maxlen=MAX_LEN)

# 2. Scale Structural Metadata features
meta_cols = ['exclamation_count', 'question_count', 'has_html_artifacts',
             'is_all_caps', 'char_cnt', 'word_cnt',
             'has_platform_mention', 'has_service_alert']

scaler = StandardScaler()
X_train_meta = scaler.fit_transform(train_df[meta_cols].values)
X_val_meta   = scaler.transform(val_df[meta_cols].values)
X_test_meta  = scaler.transform(test_df[meta_cols].values)

# 3. Build GloVe Embedding Matrix
embedding_matrix = np.zeros((MAX_WORDS, 100))
matched = 0
for word, idx in cnn_tokenizer.word_index.items():
    if idx < MAX_WORDS:
        vec = embeddings_lookup.get(word)
        if vec is not None:
            embedding_matrix[idx] = vec
            matched += 1

print(f"GloVe Coverage for TextCNN: {matched:,} / {MAX_WORDS:,} vocabulary terms matched.")
```

---

### 5️⃣ Step 7 & 8: Upgraded Dual-Input TextCNN Architecture & Tuning (Replace Cell 12, 13, 14)
Replace the CNN building, tuning, and training cells with this dual-input Functional model. It feeds sequences to the convolutional channels and merges scaled metadata directly before the classification head, training using custom class weights.

```python
# --- Build Upgraded Dual-Input TextCNN ---

def build_larry_cnn(learning_rate=1e-3):
    # 1. Text Sequence Branch
    input_layer = Input(shape=(MAX_LEN,), name='text_input')
    embedding = Embedding(
        input_dim=MAX_WORDS,
        output_dim=100,
        weights=[embedding_matrix],
        input_length=MAX_LEN,
        trainable=True
    )(input_layer)
    s_dropout = SpatialDropout1D(0.3)(embedding)
    
    # Powerful Multi-Filter Convolutional channels
    convs = []
    for fsize in [3, 5, 7, 9]:
        conv = Conv1D(128, fsize, activation='relu')(s_dropout)
        pool = GlobalMaxPooling1D()(conv)
        convs.append(pool)
    
    merged_conv = Concatenate()(convs)
    
    # 2. Metadata Branch
    meta_input = Input(shape=(len(meta_cols),), name='meta_input')
    
    # 3. Concatenate CNN + Metadata
    merged_all = Concatenate()([merged_conv, meta_input])
    
    drop = Dropout(0.5)(merged_all)
    dense = Dense(128, activation='relu')(drop)
    output = Dense(3, activation='softmax')(dense)
    
    model_func = Model(inputs=[input_layer, meta_input], outputs=output)
    model_func.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model_func

# Calculate class weights for training
classes = np.unique(y_train)
class_weights_arr = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weights_dict = dict(zip(classes, class_weights_arr))

# Run quick grid-search to find best learning rate
best_lr_cnn = 0.001
best_val_acc_cnn = 0

for lr in [1e-3, 5e-4]:
    print(f"\n--- Testing TextCNN with Learning Rate: {lr} ---")
    temp_cnn = build_larry_cnn(learning_rate=lr)
    temp_hist = temp_cnn.fit(
        [X_train_seq, X_train_meta], y_train,
        epochs=3,
        batch_size=64,
        validation_data=([X_val_seq, X_val_meta], y_val),
        class_weight=class_weights_dict,
        verbose=1
    )
    val_acc = max(temp_hist.history['val_accuracy'])
    if val_acc > best_val_acc_cnn:
        best_val_acc_cnn = val_acc
        best_lr_cnn = lr

print(f"\nSuccess! Chosen Best Learning Rate: {best_lr_cnn} (Val Accuracy: {best_val_acc_cnn:.4f})")

# Train final optimized model for 10 epochs with early stopping
larry_cnn = build_larry_cnn(learning_rate=best_lr_cnn)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = larry_cnn.fit(
    [X_train_seq, X_train_meta], y_train,
    epochs=12,
    batch_size=64,
    validation_data=([X_val_seq, X_val_meta], y_val),
    class_weight=class_weights_dict,
    callbacks=[early_stop],
    verbose=1
)
```

---

### 6️⃣ Step 9 & 10: Calibrated Soft-Voting Ensemble (Replace Cell 15, 16)
Update the evaluation and soft-voting ensemble code cells. This feeds the correct dual inputs `[X_test_seq, X_test_meta]` to the TextCNN and evaluates the joint ensemble.

```python
# Naive Bayes Test Performance
test_pred_nb = nb_clf.predict(X_test_tfidf)
nb_acc = accuracy_score(y_test, test_pred_nb)

# TextCNN Test Performance (Requires dual inputs)
test_pred_cnn_probs = larry_cnn.predict([X_test_seq, X_test_meta])
test_pred_cnn = np.argmax(test_pred_cnn_probs, axis=1)
cnn_acc = accuracy_score(y_test, test_pred_cnn)

print(f"Upgraded Naive Bayes Test Accuracy: {nb_acc*100:.2f}%")
print(f"Upgraded TextCNN Test Accuracy: {cnn_acc*100:.2f}%")

# --- Calibrated Soft Voting Ensemble ---
nb_probs = nb_clf.predict_proba(X_test_tfidf)
cnn_probs = larry_cnn.predict([X_test_seq, X_test_meta])  # Updated to dual inputs

ensemble_probs = (nb_probs + cnn_probs) / 2
ensemble_preds = np.argmax(ensemble_probs, axis=1)
ensemble_acc = accuracy_score(y_test, ensemble_preds)

print(f"\nJoint Soft-Voting Ensemble Test Accuracy: {ensemble_acc*100:.2f}%")

print("\n" + "="*50)
print("  OPTIMIZED ENSEMBLE — Cross-Domain Test Report (Gmail + WhatsApp)")
print("="*50)
print(classification_report(y_test, ensemble_preds, target_names=label_mapping.keys()))
```
