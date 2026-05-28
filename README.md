# Cross-Domain Sentiment Analysis: Mitigating Domain Shift

## 1. Project Overview
This project tackles **Domain Shift** in Sentiment Analysis. Sentiment models trained heavily on one type of text (e.g., formal reviews or short tweets) often fail when tested on drastically different text (e.g., informal WhatsApp chats, or structured emails) because the vocabulary, slang, length, and emotional expressions change drastically between domains.

### Goal
Our goal is to build an ensemble text classification pipeline that is **resilient to domain shift**. The pipeline dynamically learns how to classify text into `Positive`, `Neutral`, or `Negative` sentiment categories across various data sources including base Twitter datasets, and real-world WhatsApp messages and Gmail threads.

---

## 2. Project Architecture and Workflow
The project is built entirely within the `GROUP- K -Sentiment-Analysis-Project-With-Domain-Shift.ipynb` notebook and follows this structured machine-learning lifecycle:

### A. Data Gathering & Integration
We merged multiple source datasets ranging from formal to informal structures into a singular dataframe, retaining the specific raw text and mapping sentiment indicators to standard numeric mappings.
- **Training Corpus (The "Source" Domain):** 3 independent Twitter sentiment datasets capturing general internet colloquialisms and basic reactions.
- **Testing Corpus (The "Target" Domain):** Brand new, unseen text from WhatsApp and Gmail simulating real-world inference targets with long-form sequences and new structural habits.

### B. Cleaning & Preprocessing Workflow
The internet is messy. We built a comprehensive text-cleaning pipeline to standardise the data:
- Converting to lowercase, removing non-ASCII characters, HTML tags, and URLs.
- Replacing repeated punctuation (e.g. "???" or "!!!!") and standardizing elongated words (e.g. "sooooo" -> "so").
- Normalising numbers, usernames (`@user`), and removing excess whitespace.
- Extracting powerful metadata prior to cleaning (character count, word count, total capital letters, special character counts).

### C. Class Balancing & Data Subsampling
Sentiment datasets naturally skew towards positive and negative extremes, neglecting neutrals.
- We perform stratified subsampling to establish perfectly balanced 3-way classes for our final training set (e.g., preserving equal parts positive, neutral, negative).
- This ensures models don't take "shortcuts" by guessing the majority class and prevents skewed bias towards large internet datasets.

### D. Multi-Model 5-Fold Cross Validation
Because a single model might have blind spots on specific domains, we orchestrated an out-of-fold (OOF) 5-fold cross-validation engine encompassing 10 highly distinct models.
- **Traditional Baseline Models (TF-IDF):** Naive Bayes & Logistic Regression.
- **Classical ML Models (TF-IDF):** Support Vector Machine (LinearSVC) & Random Forest.
- **Feed-Forward Deep Learning (TF-IDF):** Multi-Layer Perceptron (MLP).
- **Sequential/Temporal Deep Learning (Embeddings):** CNN (1D text convolutions), LSTM, Bi-LSTM, and Bi-GRU to track text sequentially forward and backward.
- **Heavyweight Transformer:** DistilBERT, fine-tuned to capture contextual emotional cues across varying text.

### E. XGBoost Meta-Ensemble
Instead of blindly averaging model results or picking the single highest accuracy, we implement **Model Stacking**:
We concatenate the Out-Of-Fold probabilities of all 10 models alongside the meta-features (word counts, punctuation density, etc.) into a "meta-dataset." By training an **XGBoost Classifier** on top of this, our model autonomously learns *which* base models to trust under *which* conditions.

### F. Explainability (SHAP & LIME)
Black-box AI isn't helpful without knowing *why*.
- **SHAP (SHapley Additive exPlanations):** Validates our meta-features, visualising exactly how much "character count" or "DistilBERT confidence" influenced the XGBoost ensembler globally.
- **LIME (Local Interpretable Model-Agnostic Explanations):** Used locally to break down specific text strings so that we can see exactly which words swung the sentiment scores.

---

## 3. Libraries & Requirements
- **Data Manipulation:** `pandas`, `numpy`
- **Machine Learning / NLP:** `scikit-learn`
- **Deep Learning:** `tensorflow`, `torch`
- **Transformers:** `transformers` (HuggingFace)
- **Ensemble:** `xgboost`
- **Explainability:** `shap`, `lime`
- **Visualisation:** `matplotlib`, `seaborn`

---

## 4. How to Run
Ensure all libraries above are installed.
Open and run `GROUP- K -Sentiment-Analysis-Project-With-Domain-Shift.ipynb` sequentially from top to bottom.
