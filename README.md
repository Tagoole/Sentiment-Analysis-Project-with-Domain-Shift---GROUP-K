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

This project investigates **cross-domain sentiment generalisation** by building a unified sentiment dataset from multiple raw sources and standardising all labels into three classes: `Positive`, `Neutral`, and `Negative`.

The current repository state focuses on the full preprocessing and balancing pipeline in one notebook: `notebooks/data_cleaning.ipynb`. The cleaned dataset is split into training and validation sets (70/30, stratified by sentiment) and saved in `data/processed/`.

The central goal is to produce a high-quality, well-documented dataset that supports fair model comparison and reliable hyperparameter tuning.

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
│   │   ├── sentimentdataset.csv          ← Kaggle: Social Media Sentiment Dataset
│   │   ├── Tweets.csv                    ← Kaggle: Twitter US Airline Sentiment
│   │   └── twitterdataset.csv            ← Kaggle: Twitter Entity Sentiment Analysis
│   └── processed/
│       ├── processed_training_dataset.csv   ← 70% stratified training split
│       └── processed_validation_datset.csv  ← 30% stratified validation split
│
├── notebooks/
│   └── 00_data_cleaning.ipynb              ← Full cleaning, visualisation, balancing, and split pipeline
│
└── report/
    └── Group_K_Report.pdf           ← Final written report
```

---

## Datasets

### Training Data (merged)

| Dataset | Source | Labels |
|---|---|---|
| Social Media Sentiment | [Kaggle – kashishparmar02](https://www.kaggle.com/datasets/kashishparmar02/social-media-sentiments-analysis-dataset) | Multiple emotions mapped to 3 classes |
| Twitter US Airline Sentiment | [Kaggle – Crowdflower](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment) | positive / negative / neutral |
| Twitter Entity Sentiment Analysis | [Kaggle – jp797498e](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis) | positive / negative / neutral (+ other labels filtered during normalisation) |

After cleaning, label normalisation, deduplication, and balancing, the final combined dataset is split into:

- `processed_training_dataset.csv` (70%)
- `processed_validation_datset.csv` (30%)

Both splits keep the same sentiment proportions via stratified sampling.

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
Raw Data (3 datasets)
        ↓
Data Cleaning & EDA             [notebooks/data_cleaning.ipynb]
• Load 3 raw datasets
• Standardise to text + sentiment schema
• Handle missing/empty rows
• Normalise sentiment labels to Positive/Neutral/Negative
• Remove duplicates (within-source and cross-source)
• Downsample Negative class by 5000 samples
• Generate visualisations (before merge, after merge, balancing effect, text length, source contribution)
        ↓
Combined balanced dataset
        ↓
Stratified split (70/30)
        ↓
processed_training_dataset.csv + processed_validation_datset.csv
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
| **David** | Modelling | BERT, DistilBERT |
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

Download the three datasets from Kaggle and place them in `data/raw/`:

```
data/raw/sentimentdataset.csv
data/raw/Tweets.csv
data/raw/twitterdataset.csv
```

**Step 3 — Run data cleaning first**

Open and run `notebooks/data_cleaning.ipynb` from top to bottom. This produces:

- `data/processed/processed_training_dataset.csv`
- `data/processed/processed_validation_datset.csv`

These files are the final cleaned inputs for model training and hyperparameter validation.

---

*Group K — Makerere University, Department of Computer Science*
*Machine Learning Course — 2026*
