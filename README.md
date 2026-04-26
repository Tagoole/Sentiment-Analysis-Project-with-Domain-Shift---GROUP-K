# Group-K-Machine-Learning


# Sentiment Analysis — Group Algorithm Split

| Name | Algorithms |
|---|---|
| **David** | BERT, DistilBERT |
| **Larry** | Naive Bayes, TextCNN |
| **Ritah** | Logistic Regression, LSTM |
| **Ivy** | SVM, Bi-LSTM |
| **Julianah** | Random Forest, GRU |



# Sentiment Analysis Across Domains
### Group K — Machine Learning | Makerere University

> *Can a model trained on social media understand how people feel in emails, airline complaints, and everyday conversation? This project finds out.*

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Research Motivation](#research-motivation)
3. [Repository Structure](#repository-structure)
4. [Datasets](#datasets)
5. [Methodology](#methodology)
6. [Algorithms](#algorithms)
7. [Evaluation Strategy](#evaluation-strategy)
8. [Team](#team)
9. [Requirements](#requirements)
10. [How to Run](#how-to-run)

---

## Project Overview

This project investigates **cross-domain sentiment generalisation** — the ability of a model trained on one type of text (social media posts) to accurately classify sentiment in a different domain (emails, airline reviews, and locally collected text).

We train ten different sentiment classification models on a merged, cleaned dataset of social media posts and airline tweets, then evaluate each model against a **custom-labelled test set** assembled by our group. This test set contains text that the models have never seen and that comes from a completely different context than the training data.

The central question we are answering is not just *"which model is most accurate?"* but rather *"which model generalises best, and why?"*

---

## Research Motivation

Sentiment analysis has been widely studied. What makes this project distinctive is the evaluation design. Most published work evaluates models on held-out splits of the same dataset they were trained on — meaning the test text looks very similar to the training text. This overstates real-world performance.

In practice, a deployed sentiment model will encounter text it was never trained on: formal emails, customer complaints, mixed-language text, and platform-specific writing styles. By deliberately creating a test set from a different domain, we expose the generalisation gap that standard evaluation hides.

This approach is directly relevant to Ugandan and East African NLP contexts, where most available training data originates from Western social media platforms, yet deployed models are expected to understand local communication styles, code-switched text, and professional correspondence.

---

## Repository Structure

```
Group-K-Machine-Learning/
│
├── README.md                        ← You are here
│
├── data/
│   ├── raw/
│   │   ├── sentimentdataset.csv     ← Kaggle: Social Media Sentiment Dataset
│   │   └── Tweets.csv               ← Kaggle: Twitter US Airline Sentiment
│   └── processed/
│       ├── cleaned_train.csv        ← Final merged + cleaned training data
│       └── custom_test.csv          ← Group-labelled cross-domain test set
│
├── notebooks/
│   ├── 00_data_cleaning.ipynb       ← David: Full cleaning + EDA pipeline
│   ├── 01_naive_bayes_textcnn.ipynb ← Larry: Naive Bayes + TextCNN
│   ├── 02_logreg_lstm.ipynb         ← Ritah: Logistic Regression + LSTM
│   ├── 03_svm_bilstm.ipynb          ← Ivy: SVM + Bidirectional LSTM
│   ├── 04_randomforest_gru.ipynb    ← Julianah: Random Forest + GRU
│   ├── 05_bert_distilbert.ipynb     ← David: BERT + DistilBERT
│   └── 06_ensemble_comparison.ipynb ← All: Final ensemble + results
│
└── report/
    └── Group_K_Report.pdf           ← Final written report
```

---

## Datasets

### Training Data (merged)

| Dataset | Source | Original Size | Labels |
|---|---|---|---|
| Social Media Sentiment | [Kaggle – kashishparmar02](https://www.kaggle.com/datasets/kashishparmar02/social-media-sentiments-analysis-dataset) | 732 rows | 279 raw emotion labels → mapped to 3 classes |
| Twitter US Airline Sentiment | [Kaggle – Crowdflower](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment) | 14,640 rows | positive / negative / neutral |

After cleaning, label normalisation, deduplication, and class balancing, the final training set contains approximately **15,000 rows** with three balanced classes: `Positive`, `Negative`, `Neutral`.

### Custom Test Set (group-labelled)

To evaluate cross-domain generalisation, we manually assembled and labelled a test set from sources outside the training distribution. Each member of the group contributed samples and participated in the labelling process using agreed annotation guidelines (see report for inter-annotator agreement details).

| Source | Description |
|---|---|
| Personal emails | Professional correspondence from group members |
| Enron Email Dataset | Formal corporate email text |
| App store reviews | Google Play and App Store user reviews |
| WhatsApp messages | Informal everyday communication |

All samples were independently labelled by at least two group members. Disagreements were resolved by majority vote. Final test set size: **400–500 samples**.

---

## Methodology

```
Raw Data (2 datasets)
        ↓
  Data Cleaning & EDA          [00_data_cleaning.ipynb]
  • Drop irrelevant columns
  • Normalise 279 labels → 3 classes
  • Remove duplicates
  • Balance classes
  • Visualise distributions
        ↓
  cleaned_train.csv  ──────────────────────────────────┐
        ↓                                               │
  Individual Model Training                             │
  (5 notebooks, 2 models each)                          │
        ↓                                               │
  Predictions saved as .npy files                       │
  Results saved as .json files                          │
        ↓                                               ↓
  Final Ensemble Notebook       ←────────── custom_test.csv
  • Load all 10 model predictions
  • Weighted voting ensemble
  • In-domain vs cross-domain comparison
  • Performance gap analysis
```

---

## Algorithms

We implement ten algorithms spanning three generations of NLP approaches. Each group member trains one classical and one deep learning model, allowing a fair comparison across experience levels and computational requirements.

### Classical Machine Learning

These models use **TF-IDF** (Term Frequency–Inverse Document Frequency) to convert text into numerical feature vectors before classification.

| Algorithm | Trained By | Key Characteristic |
|---|---|---|
| **Naive Bayes** | Larry | Probabilistic; fast; strong baseline for text classification |
| **Logistic Regression** | Ritah | Linear; interpretable via word weights; often competitive with neural networks on small data |
| **Support Vector Machine (SVM)** | Ivy | Finds optimal decision boundary; historically the best pre-deep-learning text classifier |
| **Random Forest** | Julianah | Ensemble of decision trees; provides feature importance scores |

### Deep Learning

These models use **word embeddings** and sequential or convolutional architectures to capture meaning beyond individual word frequencies.

| Algorithm | Trained By | Key Characteristic |
|---|---|---|
| **TextCNN** | Larry | Convolutional filters over word sequences; captures local n-gram patterns; landmark Kim (2014) architecture |
| **LSTM** | Ritah | Reads text sequentially; maintains memory of earlier words; handles long-range dependencies |
| **Bidirectional LSTM** | Ivy | Reads text in both directions simultaneously; richer context than standard LSTM |
| **GRU** | Julianah | Simplified LSTM with fewer parameters; faster training; comparable performance |

### Transformer Models (Pre-trained)

These models leverage **transfer learning** — they were pre-trained on billions of words and fine-tuned on our dataset. They represent the current state of the art in NLP.

| Algorithm | Trained By | Key Characteristic |
|---|---|---|
| **BERT** (bert-base-uncased) | David | Google's bidirectional transformer; reads entire sentence at once using self-attention |
| **DistilBERT** (distilbert-base-uncased) | David | Compressed version of BERT; 40% fewer parameters, 60% faster, retains ~97% of performance |

### Ensemble

The final notebook combines all ten models using **weighted voting** — each model's vote is weighted by its F1 score on the validation set. This typically outperforms any single model and forms the eleventh result in our comparison.

---

## Evaluation Strategy

Each model is evaluated on two separate test sets, and both results are reported:

**1. In-domain test split** — a held-out 15% of `cleaned_train.csv`. Text that looks like the training data. This is standard evaluation.

**2. Cross-domain test set** — our `custom_test.csv`. Text from emails, app reviews, and everyday messages. This is the novel evaluation that reveals generalisation ability.

Metrics reported for every model:

| Metric | Why it matters |
|---|---|
| Accuracy | Overall correctness |
| Precision | How many predicted positives are actually positive |
| Recall | How many actual positives were caught |
| F1 Score | Harmonic mean of precision and recall — the primary comparison metric |
| Confusion Matrix | Where exactly each model makes mistakes |

The **performance gap** between in-domain and cross-domain F1 is our primary research finding. A model that drops 5% is better than one that drops 20%, even if it scored slightly lower on the standard test.

---

## Team

| Name | Role | Algorithms |
|---|---|---|
| **David** | Data Lead + Modelling | BERT, DistilBERT |
| **Larry** | Modelling | Naive Bayes, TextCNN |
| **Ritah** | Modelling | Logistic Regression, LSTM |
| **Ivy** | Modelling | SVM, Bidirectional LSTM |
| **Julianah** | Modelling | Random Forest, GRU |

---

## Requirements

```
Python 3.9+

pandas
numpy
scikit-learn
matplotlib
seaborn
tensorflow >= 2.10
keras
transformers >= 4.30
torch
joblib
scipy
```

Install all dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn tensorflow keras transformers torch joblib scipy
```

For BERT and DistilBERT notebooks, a GPU is strongly recommended. Use **Google Colab** (free T4 GPU) if no local GPU is available.

---

## How to Run

**Step 1 — Clone the repository**

```bash
git clone https://github.com/your-username/Group-K-Machine-Learning.git
cd Group-K-Machine-Learning
```

**Step 2 — Place raw data files**

Download the two datasets from Kaggle and place them in `data/raw/`:

```
data/raw/sentimentdataset.csv
data/raw/Tweets.csv
```

**Step 3 — Run data cleaning first**

Open and run `notebooks/00_data_cleaning.ipynb` completely before any other notebook. This produces `data/processed/cleaned_train.csv` which every other notebook depends on.

**Step 4 — Run individual model notebooks**

Each member runs their own notebook independently. The notebooks are self-contained and only require `cleaned_train.csv` as input.

**Step 5 — Run the final ensemble notebook**

Once all five training notebooks have been run and predictions saved, open `notebooks/06_ensemble_comparison.ipynb` to see the full comparison table, ensemble results, and cross-domain analysis.

---

*Group K — Makerere University, Department of Computer Science*
*Machine Learning Course — 2026*
