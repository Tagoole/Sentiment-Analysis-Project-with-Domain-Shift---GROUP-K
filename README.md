# Cross-Domain Sentiment Analysis: Mitigating Domain Shift

## 1. Project Overview
This notebook project evaluates how well an ensemble sentiment classifier trained on Twitter-style text generalises to structurally different domains: Gmail emails, WhatsApp chats, and App Reviews.

The core problem is domain shift: a model trained on one text style can lose accuracy when the new data uses different vocabulary, formatting, length, and tone.

### Goal
Build a robust sentiment stack that classifies `Positive`, `Neutral`, and `Negative` sentiment across source and target domains while explicitly measuring the cross-domain performance drop.

---

## 2. Project Architecture and Workflow
The implementation lives in `notebooks/GROUP K Sentiment-Analysis-Project-With-Domain-Shift.ipynb` and follows a reproducible ML pipeline.

### A. Data Loading
The notebook loads:
- source training data from processed Twitter sentiment datasets
- test domains from `data/processed/gmail_test_data.csv`, `data/processed/whatsapp_test_data.csv`, and `data/processed/labelled_app_reviews_test.csv`

### B. Metadata and Text Preprocessing
The notebook extracts metadata features such as exclamation/question counts, all-caps flags, character/word counts, platform mentions, service-alert signals, and positive/negative cue counts.

Text cleaning uses the notebook's `surgical_cleaner`, which:
- lowercases text
- removes URLs
- removes punctuation characters such as `,.;:()[]{}<>/@#`
- normalises whitespace
- replaces empty output with `"notification"`

### C. Feature Engineering
The notebook builds:
- TF-IDF features for classical models
- token sequences and GloVe-based embeddings for deep learning models
- scaled metadata features for the stacked ensemble

### D. Model Training
A 5-fold out-of-fold training scheme is used across 10 base learners:
- Naive Bayes
- Logistic Regression
- Linear SVC (with calibration)
- Random Forest
- Multi-Layer Perceptron
- CNN
- LSTM
- Bi-LSTM
- Bi-GRU
- DistilBERT

The notebook collects probability outputs from each model for stacking.

### E. Stacking and Adaptation
A stacked XGBoost meta-learner is trained on the base-model probabilities plus meta-features.

The notebook also adapts this stack using pseudo-labeling:
- high-confidence target-domain predictions are pseudo-labeled
- these examples are added to the stacked training data
- a second adapted XGBoost meta-learner is trained for the final domain evaluation

### F. Evaluation and Explainability
The notebook produces:
- domain-specific classification reports
- confusion matrices for Gmail, WhatsApp, and App Reviews
- ROC curves for App Reviews
- SHAP explanations for the XGBoost meta-model
- LIME explanations for specific text predictions

---

## 3. Key Results
The notebook reports the following final domain performance after stacking and adaptation:

| Domain | Accuracy | Weighted F1 |
|---|---|---|
| Gmail | 50.86% | 0.50 |
| WhatsApp | 61.64% | 0.63 |
| App Reviews | 54.11% | 0.51 |

The notebook also highlights that source-domain OOF training accuracy was around **88.3%**, showing how domain shift reduces performance on unseen Gmail, WhatsApp, and App Reviews data.

---

## 4. Libraries & Requirements
- **Data Manipulation:** `pandas`, `numpy`
- **Machine Learning / NLP:** `scikit-learn`
- **Deep Learning:** `tensorflow`, `torch`
- **Transformers:** `transformers` (HuggingFace)
- **Ensemble:** `xgboost`
- **Explainability:** `shap`, `lime`
- **Visualisation:** `matplotlib`, `seaborn`

---

## 5. How to Run
Install the required libraries.
Open `notebooks/GROUP K Sentiment-Analysis-Project-With-Domain-Shift.ipynb` and execute the notebook cells from top to bottom.
