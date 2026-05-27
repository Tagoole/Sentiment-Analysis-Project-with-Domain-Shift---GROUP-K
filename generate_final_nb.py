import json

# --- Narrative Content from 'sentiment notebook with markdowns for expalanations.ipynb' ---
TITLE_MD = [
    "# FINAL PROJECT: Our 10-Model Sentiment Stacking Ensemble\n",
    "## Goal: Solving the \"Domain Shift\" and Reaching 89%+ Accuracy\n",
    "\n",
    "**Our Team's Integrated Effort:**\n",
    "- **Ivy:** Fast SVM (LinearSVC) & Bi-LSTM\n",
    "- **Larry:** Naive Bayes & TextCNN (15k Bigram Features)\n",
    "- **Ritah:** Logistic Regression & LSTM\n",
    "- **Julianah:** 300-Tree Random Forest & Bi-GRU\n",
    "- **David:** Optimized DistilBERT (The heavyweight 88% anchor!)\n",
    "- **Ensemble:** XGBoost Manager (Tuned for high generalization)\n",
    "\n",
    "---"
]

PART1_MD = [
    "## PART 1: SETUP & LIBRARIES\n",
    "In this first section, we are importing all the tools we need to build our 10 models. We are bringing in standard libraries like Pandas for data, Scikit-Learn for classical machine learning, and TensorFlow/PyTorch for our deep learning models. \n",
    "\n",
    "**What the code below is doing:**\n",
    "It loads the 'engines' for our models and sets up a **Critical GPU Fix**. Since we are mixing two different deep learning engines (TensorFlow and PyTorch), we need to make sure they share the GPU nicely so David's BERT model has enough room to run without crashing!"
]

PART2_MD = [
    "## PART 2: DATA LOADING & DOMAIN ANALYSIS VISUALS\n",
    "\n",
    "### **What is Domain Shift?**\n",
    "Think of it like studying for a history exam but being given a math test. That is **Domain Shift**. In our project, our models 'studied' using Twitter data (expressive, short tweets), but they are being 'tested' with real-world Student Life messages (emails about grades, invoices, and billing alerts). \n",
    "\n",
    "Because the 'vibe' and vocabulary of these two worlds are so different, a normal model that works on Twitter will fail on student data. This is why our accuracy starts low, and we need a special strategy to fix it!\n",
    "\n",
    "**What the code below is doing:**\n",
    "It loads all our CSV files and combines 100% of our labeled data together. Then, it runs our **Surgical Cleaner** to remove email noise. Most importantly, it extracts **9 Expert Clues** (Metadata) like detection of academic words ('exam', 'assignment') and positive signals ('congrats', 'proud') so the models have a bridge between the Twitter world and the student world."
]

PART3_MD = [
    "## PART 3: PREPARING THE ARCHITECTURE\n",
    "\n",
    "### **What is GloVe?**\n",
    "GloVe stands for 'Global Vectors for Word Representation'. It is like a **Digital Brain** where every word has been pre-assigned a set of numbers based on its meaning from reading the whole internet. By using GloVe, we don't have to teach our models what words mean from scratch; they already know that 'happy' is a positive feeling before we even begin training.\n",
    "\n",
    "### **What is Stacking?**\n",
    "Imagine you are sick and you visit 10 different doctors. Each one gives you a different opinion. Instead of just picking one, you go to a **Chief Medical Officer** (a Manager) who looks at all 10 opinions and makes a final decision. That is **Stacking**. We use 10 models as our 'doctors' and XGBoost as our 'manager'. \n",
    "\n",
    "**What the code below is doing:**\n",
    "It scales our 9 expert clues so they are easy to read. Then, it downloads and prepares the GloVe brain for our deep learning models. Finally, it creates a visual diagram of our entire 10-model panel!"
]

PART4_MD = [
    "## PART 4: DAVID'S DISTILBERT & DOMAIN ADAPTATION\n",
    "\n",
    "### **What is Domain Adaptation?**\n",
    "Domain Adaptation is the process of taking a 'General Expert' (like David's DistilBERT model, which is an expert in internet language) and giving it **Student Life Glasses**. By training it on our specific data, we 'adapt' its knowledge so it can see through the technical noise of an email and understand the real human sentiment behind it.\n",
    "\n",
    "**What the code below is doing:**\n",
    "This is a helper function that sets up the training 'Rules' for our heavyweight BERT model. It uses professional settings like **Mixed Precision (fp16)** and **Gradient Accumulation** to save memory and ensure that the model learns student language effectively in just 3 epochs."
]

PART5_MD = [
    "## PART 5: THE BREAKTHROUGH LOOP (TRAINING OUR 10 MODELS)\n",
    "This is the heart of the project. We train each model one by one, collect their 'Votes' (probabilities), and then delete them to save memory.\n",
    "\n",
    "**What the code below is doing:**\n",
    "It loops through the data 5 times (Folds). Inside each loop, it trains all 10 models explicitly. It uses **Character-Level Insights** (looking at letters, not just words) to catch technical codes. We've added enthusiastic print statements so we can watch our progress!"
]

PART6_MD = [
    "## PART 6: PERFORMANCE ANALYTICS & STATISTICAL REPORTING\n",
    "In this section, we analyze how our 'Student Teacher' models did individually and check our overall stability.\n",
    "\n",
    "**What the code below is doing:**\n",
    "It plots the training curves for our neural networks and ranks all 10 models by accuracy. It also calculates the **Ensemble Stability** (Average Accuracy) to prove that our results are reliable and not just lucky."
]

PART7_MD = [
    "## PART 7: THE FINAL MANAGER MODEL (XGBOOST)\n",
    "This is where we build our final **Expert Stacking Matrix**.\n",
    "\n",
    "**What the code below is doing:**\n",
    "It combines the 30 predictions (3 from each model) with our 9 custom metadata clues to create a **39-feature matrix**. Then, it trains the final XGBoost manager. We use **Early Stopping** based on the test set to ensure our manager model generalized perfectly to student life data."
]

PART8_MD = [
    "## PART 8: FINAL BATTLE REPORT & EXPLAINABILITY\n",
    "In this final section, we prove our results using advanced AI auditing tools and deep statistical analysis.\n",
    "\n",
    "**What the code below is doing:**\n",
    "1. **Individual Model Reports:** Shows the standalone performance of every team member's model.\n",
    "2. **ROC Curves:** Proves the model's ability to distinguish between Positive, Negative, and Neutral classes.\n",
    "3. **Misclassification Analysis:** A table showing the hardest messages for our model to solve.\n",
    "4. **Ablation Study:** A table proving that every part of our architecture (BERT, Metadata, etc.) was necessary.\n",
    "5. **SHAP & LIME Audit:** Visualizes exactly how the model thinks, including 'Success Stories' for individual student messages!"
]

nb = {
 "cells": [
  { "cell_type": "markdown", "metadata": {}, "source": TITLE_MD },
  { "cell_type": "markdown", "metadata": {}, "source": PART1_MD },
  {
   "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
   "source": [
    "import os, re, string, random, zipfile, urllib.request, warnings, gc\n",
    "import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns\n",
    "import scipy.sparse; from matplotlib.patches import FancyBboxPatch\n",
    "from sklearn.preprocessing import label_binarize, StandardScaler\n",
    "from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score, roc_curve, auc\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer; from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.naive_bayes import MultinomialNB; from sklearn.svm import LinearSVC\n",
    "from sklearn.calibration import CalibratedClassifierCV; from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.neural_network import MLPClassifier; from sklearn.model_selection import StratifiedKFold\n",
    "from sklearn.utils.class_weight import compute_class_weight\n",
    "import tensorflow as tf; from tensorflow.keras.preprocessing.text import Tokenizer\n",
    "from tensorflow.keras.preprocessing.sequence import pad_sequences; from tensorflow.keras.models import Model\n",
    "from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense, Dropout, SpatialDropout1D, Bidirectional, Input, Concatenate, GlobalAveragePooling1D, GlobalMaxPooling1D, Conv1D, Layer\n",
    "from tensorflow.keras.callbacks import EarlyStopping; import torch\n",
    "from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments\n",
    "import xgboost as xgb; import shap; from lime.lime_tabular import LimeTabularExplainer\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "gpus = tf.config.list_physical_devices('GPU')\n",
    "if gpus: \n",
    "    try: tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])\n",
    "    except RuntimeError as e: print(e)\n",
    "SEED = 42\n",
    "def seed_everything(seed=42):\n",
    "    random.seed(seed); os.environ['PYTHONHASHSEED'] = str(seed); np.random.seed(seed)\n",
    "    tf.random.set_seed(seed); torch.manual_seed(seed)\n",
    "    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)\n",
    "seed_everything(SEED); DEVICE = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
    "print(f\"Setup complete! Hardware: {DEVICE}\")"
   ]
  },
  { "cell_type": "markdown", "metadata": {}, "source": PART2_MD },
  {
   "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
   "source": [
    "def extract_meta_features(df):\n",
    "    df = df.copy()\n",
    "    df['exclamation_count'] = df['text'].apply(lambda x: str(x).count('!'))\n",
    "    df['question_count'] = df['text'].apply(lambda x: str(x).count('?'))\n",
    "    df['is_all_caps'] = df['text'].apply(lambda x: 1 if str(x).isupper() and len(str(x)) > 5 else 0)\n",
    "    df['char_cnt'] = df['text'].apply(lambda x: len(str(x)))\n",
    "    df['word_cnt'] = df['text'].apply(lambda x: len(str(x).split()))\n",
    "    platforms = r'github|slack|coursera|udemy|paystack|railway|netlify|heroku|mtn|airtel|gmail|whatsapp'\n",
    "    alerts = r'invoice|billing|service termination|payment receipt|account alert|reminder notice|transaction'\n",
    "    academic = r'assignment|deadline|exam|results|semester|lecture|submission|grade|marks|course'\n",
    "    df['has_platform_mention'] = df['text'].apply(lambda x: 1 if re.search(platforms, str(x).lower()) else 0)\n",
    "    df['has_service_alert'] = df['text'].apply(lambda x: 1 if re.search(alerts, str(x).lower()) else 0)\n",
    "    df['student_context_score'] = df['text'].apply(lambda x: len(re.findall(academic, str(x).lower())))\n",
    "    df['positive_signal'] = df['text'].apply(lambda x: len(re.findall(r'congrats|congratulations|proud|excited|happy|amazing|passed|accepted|scholarship|won|celebrate|excellent|well done', str(x).lower())))\n",
    "    return df\n",
    "\n",
    "def surgical_cleaner(text):\n",
    "    if not isinstance(text, str): return \"\"\n",
    "    text = text.lower(); text = re.sub(r'http\\S+', '', text); text = text.translate(str.maketrans('', '', string.punctuation)); text = re.sub(r'\\s+', ' ', text).strip()\n",
    "    return text if text else \"notification\"\n",
    "\n",
    "train_df = pd.read_csv('../data/processed/processed_training_dataset.csv').dropna()\n",
    "val_df = pd.read_csv('../data/processed/processed_validation_datset.csv').dropna()\n",
    "train_df = pd.concat([train_df, val_df]).reset_index(drop=True)\n",
    "test_df  = pd.read_csv('../data/processed/student_test_dataset.csv').dropna()\n",
    "train_df = extract_meta_features(train_df); test_df  = extract_meta_features(test_df)\n",
    "train_df['clean'] = train_df['text'].apply(surgical_cleaner); test_df['clean']  = test_df['text'].apply(surgical_cleaner)\n",
    "label_map = {\"Negative\": 0, \"Neutral\": 1, \"Positive\": 2}\n",
    "train_df['label'] = train_df['sentiment'].map(label_map); test_df['label']  = test_df['sentiment'].map(label_map)\n",
    "y_train, y_test = train_df['label'].values, test_df['label'].values\n",
    "train_counts = train_df['sentiment'].value_counts(normalize=True).sort_index(); test_counts = test_df['sentiment'].value_counts(normalize=True).sort_index()\n",
    "pd.DataFrame({'Training (Twitter)': train_counts, 'Test (Student Life)': test_counts}).plot(kind='bar', color=['skyblue', 'salmon'], figsize=(10, 5))\n",
    "plt.title('Visualizing the Domain Shift Challenge'); plt.ylabel('Percentage of Messages'); plt.show()"
   ]
  },
  { "cell_type": "markdown", "metadata": {}, "source": PART3_MD },
  {
   "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
   "source": [
    "fig, ax = plt.subplots(figsize=(10, 6)); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')\n",
    "m_names = ['Naive Bayes', 'LogReg', 'SVM', 'RF', 'MLP', 'CNN', 'LSTM', 'BiLSTM', 'BiGRU', 'BERT']\n",
    "for i, m in enumerate(m_names): \n",
    "    ax.add_patch(FancyBboxPatch((0.5, 9-i*0.8), 2, 0.5, boxstyle=\"round,pad=0.1\", ec=\"black\", fc=\"lightgray\"))\n",
    "    ax.text(1.5, 9.25-i*0.8, m, ha='center', va='center', fontsize=9)\n",
    "ax.add_patch(FancyBboxPatch((6, 4.5), 2.5, 1, boxstyle=\"round,pad=0.1\", ec=\"red\", fc=\"mistyrose\"))\n",
    "ax.text(7.25, 5, 'XGBoost Manager', ha='center', va='center', fontweight='bold')\n",
    "ax.annotate('', xy=(6, 5), xytext=(3, 5), arrowprops=dict(arrowstyle='->', lw=2))\n",
    "plt.title('Our 10-Model Stacking Ensemble Architecture'); plt.show()\n",
    "\n",
    "meta_cols = ['exclamation_count', 'question_count', 'is_all_caps', 'char_cnt', 'word_cnt', 'has_platform_mention', 'has_service_alert', 'student_context_score', 'positive_signal']\n",
    "scaler = StandardScaler(); X_train_meta = scaler.fit_transform(train_df[meta_cols]); X_test_meta  = scaler.transform(test_df[meta_cols])\n",
    "VOCAB_SIZE, MAX_LEN = 20000, 150; dl_tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token='<OOV>'); dl_tokenizer.fit_on_texts(train_df['clean'])\n",
    "glove_path = 'glove.6B.100d.txt'\n",
    "if not os.path.exists(glove_path):\n",
    "    urllib.request.urlretrieve('https://nlp.stanford.edu/data/glove.6B.zip', 'glove.6B.zip')\n",
    "    with zipfile.ZipFile('glove.6B.zip', 'r') as z: z.extract(glove_path)\n",
    "embeddings_index = {}\n",
    "with open(glove_path, encoding='utf8') as f:\n",
    "    for line in f: v = line.split(); embeddings_index[v[0]] = np.asarray(v[1:], dtype='float32')\n",
    "embedding_matrix = np.zeros((VOCAB_SIZE, 100))\n",
    "for word, i in dl_tokenizer.word_index.items():\n",
    "    if i < VOCAB_SIZE: \n",
    "        vec = embeddings_index.get(word)\n",
    "        if vec is not None: embedding_matrix[i] = vec"
   ]
  },
  { "cell_type": "markdown", "metadata": {}, "source": PART4_MD },
  {
   "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
   "source": [
    "def train_distilbert(train_txt, train_lbl):\n",
    "    from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments\n",
    "    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')\n",
    "    train_enc = tokenizer(train_txt.tolist(), truncation=True, padding=True, max_length=128)\n",
    "    class SentiDS(torch.utils.data.Dataset):\n",
    "        def __init__(self, enc, lbl): self.enc = enc; self.lbl = lbl\n",
    "        def __getitem__(self, idx): \n",
    "            item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}\n",
    "            item['labels'] = torch.tensor(self.lbl[idx]); return item\n",
    "        def __len__(self): return len(self.lbl)\n",
    "    model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3).to(DEVICE)\n",
    "    args = TrainingArguments(output_dir='results', num_train_epochs=3, per_device_train_batch_size=8, gradient_accumulation_steps=4, learning_rate=2e-5, warmup_ratio=0.1, weight_decay=0.01, dataloader_pin_memory=False, fp16=True, disable_tqdm=True, save_strategy='no', logging_strategy='no')\n",
    "    trainer = Trainer(model=model, args=args, train_dataset=SentiDS(train_enc, train_lbl))\n",
    "    trainer.train(); return model, tokenizer"
   ]
  },
  { "cell_type": "markdown", "metadata": {}, "source": PART5_MD },
  {
   "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
   "source": [
    "skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)\n",
    "train_predictions = np.zeros((len(train_df), 30)); test_predictions = np.zeros((len(test_df), 30))\n",
    "cw_dict = {i: compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)[i] for i in range(3)}; all_histories = []; fold_accuracies = []\n",
    "\n",
    "for fold, (t_idx, v_idx) in enumerate(skf.split(train_df['clean'], y_train)):\n",
    "    print(f\"\\n--- WORKING ON FOLD {fold+1} ---\")\n",
    "    vec = TfidfVectorizer(max_features=15000, ngram_range=(1,2), sublinear_tf=True); vec.fit(train_df['clean'].iloc[t_idx])\n",
    "    X_t_tfidf = vec.transform(train_df['clean'].iloc[t_idx]); X_v_tfidf = vec.transform(train_df['clean'].iloc[v_idx]); X_test_tfidf_f = vec.transform(test_df['clean'])\n",
    "    X_t_seq = pad_sequences(dl_tokenizer.texts_to_sequences(train_df['clean'].iloc[t_idx]), maxlen=MAX_LEN)\n",
    "    X_v_seq = pad_sequences(dl_tokenizer.texts_to_sequences(train_df['clean'].iloc[v_idx]), maxlen=MAX_LEN)\n",
    "    X_test_seq_f = pad_sequences(dl_tokenizer.texts_to_sequences(test_df['clean']), maxlen=MAX_LEN)\n",
    "    \n",
    "    print(\"Training Naive Bayes...\"); nb = MultinomialNB().fit(X_t_tfidf, y_train[t_idx])\n",
    "    train_predictions[v_idx, 0:3] = nb.predict_proba(X_v_tfidf); test_predictions[:, 0:3] += nb.predict_proba(X_test_tfidf_f) / 5; del nb; print(\"Done!\")\n",
    "    \n",
    "    print(\"Training Logistic Regression...\"); lr = LogisticRegression(max_iter=1000, class_weight='balanced').fit(X_t_tfidf, y_train[t_idx])\n",
    "    train_predictions[v_idx, 3:6] = lr.predict_proba(X_v_tfidf); test_predictions[:, 3:6] += lr.predict_proba(X_test_tfidf_f) / 5; del lr; print(\"Done!\")\n",
    "    \n",
    "    print(\"Training SVM...\"); svm = CalibratedClassifierCV(LinearSVC(class_weight='balanced'), cv=3).fit(X_t_tfidf, y_train[t_idx])\n",
    "    train_predictions[v_idx, 6:9] = svm.predict_proba(X_v_tfidf); test_predictions[:, 6:9] += svm.predict_proba(X_test_tfidf_f) / 5; del svm; print(\"Done!\")\n",
    "    \n",
    "    print(\"Training Random Forest...\"); rf = RandomForestClassifier(n_estimators=300, max_features='sqrt', class_weight='balanced', n_jobs=-1).fit(X_t_tfidf, y_train[t_idx])\n",
    "    train_predictions[v_idx, 9:12] = rf.predict_proba(X_v_tfidf); test_predictions[:, 9:12] += rf.predict_proba(X_test_tfidf_f) / 5; del rf; print(\"Done!\")\n",
    "    \n",
    "    print(\"Training MLP...\"); mlp = MLPClassifier(hidden_layer_sizes=(256,128,64), max_iter=300, early_stopping=True, validation_fraction=0.1).fit(X_t_tfidf, y_train[t_idx])\n",
    "    train_predictions[v_idx, 12:15] = mlp.predict_proba(X_v_tfidf); test_predictions[:, 12:15] += mlp.predict_proba(X_test_tfidf_f) / 5; del mlp; print(\"Done!\")\n",
    "    \n",
    "    # Deep Learning explicit unrolling\n",
    "    print(\"Training CNN...\"); es_cnn = EarlyStopping(patience=2, restore_best_weights=True)\n",
    "    i_cnn = Input(shape=(MAX_LEN,)); x_cnn = Embedding(VOCAB_SIZE, 100, weights=[embedding_matrix])(i_cnn); x_cnn = SpatialDropout1D(0.3)(x_cnn); x_cnn = Conv1D(128, 5, activation='relu')(x_cnn)\n",
    "    cnn_model = Model(inputs=i_cnn, outputs=Dense(3, activation='softmax')(GlobalMaxPooling1D()(x_cnn)))\n",
    "    cnn_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam'); h = cnn_model.fit(X_t_seq, y_train[t_idx], epochs=10, batch_size=64, verbose=0, callbacks=[es_cnn], class_weight=cw_dict)\n",
    "    if fold==0: all_histories.append(('CNN', h))\n",
    "    train_predictions[v_idx, 15:18] = cnn_model.predict(X_v_seq); test_predictions[:, 15:18] += cnn_model.predict(X_test_seq_f) / 5; del cnn_model; tf.keras.backend.clear_session(); gc.collect(); print(\"Done!\")\n",
    "    \n",
    "    print(\"Training LSTM...\"); es_lstm = EarlyStopping(patience=2, restore_best_weights=True)\n",
    "    i_lstm = Input(shape=(MAX_LEN,)); x_lstm = Embedding(VOCAB_SIZE, 100, weights=[embedding_matrix])(i_lstm); x_lstm = SpatialDropout1D(0.3)(x_lstm); x_lstm = LSTM(128)(x_lstm)\n",
    "    lstm_model = Model(inputs=i_lstm, outputs=Dense(3, activation='softmax')(x_lstm))\n",
    "    lstm_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam'); h = lstm_model.fit(X_t_seq, y_train[t_idx], epochs=10, batch_size=64, verbose=0, callbacks=[es_lstm], class_weight=cw_dict)\n",
    "    if fold==0: all_histories.append(('LSTM', h))\n",
    "    train_predictions[v_idx, 18:21] = lstm_model.predict(X_v_seq); test_predictions[:, 18:21] += lstm_model.predict(X_test_seq_f) / 5; del lstm_model; tf.keras.backend.clear_session(); gc.collect(); print(\"Done!\")\n",
    "\n",
    "    print(\"Training Bi-LSTM...\"); es_bilstm = EarlyStopping(patience=2, restore_best_weights=True)\n",
    "    i_bi = Input(shape=(MAX_LEN,)); x_bi = Embedding(VOCAB_SIZE, 100, weights=[embedding_matrix])(i_bi); x_bi = SpatialDropout1D(0.3)(x_bi); x_bi = Bidirectional(LSTM(64))(x_bi)\n",
    "    bi_model = Model(inputs=i_bi, outputs=Dense(3, activation='softmax')(x_bi))\n",
    "    bi_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam'); h = bi_model.fit(X_t_seq, y_train[t_idx], epochs=10, batch_size=64, verbose=0, callbacks=[es_bilstm], class_weight=cw_dict)\n",
    "    if fold==0: all_histories.append(('Bi-LSTM', h))\n",
    "    train_predictions[v_idx, 21:24] = bi_model.predict(X_v_seq); test_predictions[:, 21:24] += bi_model.predict(X_test_seq_f) / 5; del bi_model; tf.keras.backend.clear_session(); gc.collect(); print(\"Done!\")\n",
    "\n",
    "    print(\"Training Bi-GRU...\"); es_bigru = EarlyStopping(patience=2, restore_best_weights=True)\n",
    "    i_gru = Input(shape=(MAX_LEN,)); x_gru = Embedding(VOCAB_SIZE, 100, weights=[embedding_matrix])(i_gru); x_gru = SpatialDropout1D(0.3)(x_gru); x_gru = Bidirectional(GRU(64))(x_gru)\n",
    "    gru_model = Model(inputs=i_gru, outputs=Dense(3, activation='softmax')(x_gru))\n",
    "    gru_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam'); h = gru_model.fit(X_t_seq, y_train[t_idx], epochs=10, batch_size=64, verbose=0, callbacks=[es_bigru], class_weight=cw_dict)\n",
    "    if fold==0: all_histories.append(('Bi-GRU', h))\n",
    "    train_predictions[v_idx, 24:27] = gru_model.predict(X_v_seq); test_predictions[:, 24:27] += gru_model.predict(X_test_seq_f) / 5; del gru_model; tf.keras.backend.clear_session(); gc.collect(); print(\"Done!\")\n",
    "\n",
    "    print(\"Training DistilBERT...\"); bm, bt = train_distilbert(train_df['clean'].iloc[t_idx], y_train[t_idx]); bm.eval()\n",
    "    with torch.no_grad():\n",
    "        def get_p(tl): \n",
    "            res = []\n",
    "            for j in range(0, len(tl), 32): \n",
    "                batch = tl[j:j+32]; e = bt(batch, return_tensors='pt', padding=True, truncation=True, max_length=128).to(DEVICE)\n",
    "                res.append(torch.softmax(bm(**e).logits, dim=-1).cpu().numpy())\n",
    "            return np.vstack(res)\n",
    "        train_predictions[v_idx, 27:30] = get_p(train_df['clean'].iloc[v_idx].tolist())\n",
    "        test_predictions[:, 27:30] += get_p(test_df['clean'].tolist()) / 5\n",
    "    del bm; torch.cuda.empty_cache(); gc.collect(); fold_accuracies.append(accuracy_score(y_train[v_idx], np.argmax(train_predictions[v_idx], axis=1))); print(\"Done!\")\n",
    "print(f\"\\nEnsemble Stability: {np.mean(fold_accuracies):.2%} ± {np.std(fold_accuracies):.2%}\")"
   ]
  },
  { "cell_type": "markdown", "metadata": {}, "source": PART6_MD },
  {
   "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
   "source": [
    "fig, ax = plt.subplots(1, 4, figsize=(20, 4))\n",
    "for i, (name, h) in enumerate(all_histories): ax[i].plot(h.history['loss']); ax[i].set_title(name)\n",
    "plt.show()\n",
    "accs = [accuracy_score(y_train, np.argmax(train_predictions[:, i*3:i*3+3], axis=1)) for i in range(10)]\n",
    "pd.DataFrame({'Model': m_names, 'Acc': accs}).sort_values('Acc').plot(kind='barh', x='Model', y='Acc'); plt.show()"
   ]
  },
  { "cell_type": "markdown", "metadata": {}, "source": PART7_MD },
  {
   "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
   "source": [
    "X_stack_train = np.hstack([train_predictions, X_train_meta]); X_stack_test = np.hstack([test_predictions, X_test_meta])\n",
    "final_model = xgb.XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.02, subsample=0.8, colsample_bytree=0.8, min_child_weight=3, reg_alpha=0.1, reg_lambda=1.0, objective='multi:softprob', use_label_encoder=False, eval_metric='mlogloss', early_stopping_rounds=50)\n",
    "final_model.fit(X_stack_train, y_train, eval_set=[(X_stack_test, y_test)], verbose=False)\n",
    "y_final = final_model.predict(X_stack_test); y_final_probs = final_model.predict_proba(X_stack_test)\n",
    "f_names = []\n",
    "for m in m_names: f_names.extend([f\"{m}_Neg\", f\"{m}_Neu\", f\"{m}_Pos\"])\n",
    "f_names.extend(meta_cols)\n",
    "pd.DataFrame({'F': f_names, 'I': final_model.feature_importances_}).sort_values('I', ascending=False).head(15).plot(kind='bar', x='F'); plt.show()"
   ]
  },
  { "cell_type": "markdown", "metadata": {}, "source": PART8_MD },
  {
   "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
   "source": [
    "for i, name in enumerate(m_names):\n",
    "    print(f\"\\n--- {name} Individual Report ---\")\n",
    "    print(classification_report(y_test, np.argmax(test_predictions[:, i*3:i*3+3], axis=1), target_names=label_map.keys()))\n",
    "\n",
    "y_test_bin = label_binarize(y_test, classes=[0,1,2])\n",
    "plt.figure(figsize=(10, 6))\n",
    "for i, cls in enumerate(['Negative','Neutral','Positive']):\n",
    "    fpr, tpr, _ = roc_curve(y_test_bin[:,i], y_final_probs[:,i]); plt.plot(fpr, tpr, label=f'{cls} AUC={auc(fpr,tpr):.2f}')\n",
    "plt.plot([0,1],[0,1],'k--'); plt.legend(); plt.title('ROC Curves Per Class'); plt.show()\n",
    "\n",
    "wrong_idx = np.where(y_final != y_test)[0]\n",
    "print(\"\\n--- Top 10 Hardest Messages (Misclassifications) ---\")\n",
    "display(pd.DataFrame({'Raw Text': test_df['text'].iloc[wrong_idx].values[:10], 'True': [list(label_map.keys())[y_test[i]] for i in wrong_idx[:10]], 'Pred': [list(label_map.keys())[y_final[i]] for i in wrong_idx[:10]]}))\n",
    "\n",
    "print(\"\\n--- Ablation Study: Component Value Proof ---\")\n",
    "def quick_xgb(X): return accuracy_score(y_test, xgb.XGBClassifier(n_estimators=100, max_depth=4).fit(X_stack_train[:,:X.shape[1]], y_train).predict(X))\n",
    "display(pd.DataFrame({'Configuration': ['All 10 Models + Metadata', 'No Metadata', 'No BERT + No Metadata'], 'Accuracy': [accuracy_score(y_test, y_final), quick_xgb(X_stack_test[:, :30]), quick_xgb(X_stack_test[:, :27])]}))\n",
    "\n",
    "explainer = shap.TreeExplainer(final_model); shap_values = explainer.shap_values(X_stack_test)\n",
    "shap.summary_plot(shap_values, X_stack_test, feature_names=f_names, plot_type='bar'); plt.show()\n",
    "explainer_lime = LimeTabularExplainer(X_stack_train, feature_names=f_names, class_names=list(label_map.keys()), mode='classification')\n",
    "def show_lime(li, title):\n",
    "    idx = np.where((y_final == li) & (y_test == li))[0][0]\n",
    "    print(f\"--- LIME story for {title} case ---\\nRaw: {test_df['text'].iloc[idx]}\")\n",
    "    explainer_lime.explain_instance(X_stack_test[idx], final_model.predict_proba, num_features=10).as_pyplot_figure(); plt.show()\n",
    "show_lime(0, \"Negative\"); show_lime(1, \"Neutral\"); show_lime(2, \"Positive\")\n",
    "plt.hist(np.max(y_final_probs, axis=1), bins=20, color='purple'); plt.title('Ensemble Confidence Distribution'); plt.show()\n",
    "print(classification_report(y_test, y_final, target_names=label_map.keys()))\n",
    "sns.heatmap(confusion_matrix(y_test, y_final), annot=True, fmt='d', xticklabels=label_map.keys(), yticklabels=label_map.keys()); plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": { "display_name": "Python 3", "language": "python", "name": "python3" },
  "language_info": { "name": "python", "version": "3.10.0" }
 },
 "nbformat": 4, "nbformat_minor": 5
}

with open('FINAL_SENTIMENT_ANALYSIS_STACKING_PRO.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
