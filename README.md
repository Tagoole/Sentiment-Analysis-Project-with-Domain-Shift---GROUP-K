# Evaluating Sentiment Model Generalization: From Social Media to Student Communications
### Assessing Performance on Real-World Gmail and WhatsApp Data

> *Can a model trained on social media understand how people feel in emails and private messages? This project evaluates model generalization to real-world student data.*

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

This project investigates **cross-domain sentiment generalization** by building a unified sentiment dataset and evaluating performance on unseen, student-centric data from Gmail and WhatsApp.

The repository includes a comprehensive preprocessing pipeline and a collection of models ranging from classical algorithms to modern transformers.

---

## Research Motivation

Sentiment analysis has been widely studied. This project focuses on the evaluation design: testing models on data that is entirely different from the training distribution. While most models perform well on similar data, their performance often drops when encountering real-world communication styles like formal emails or informal chats.

By manually creating and labeling a test set from Gmail and WhatsApp, we provide a more realistic assessment of how these models would perform in practice for students.

---

## Repository Structure

```
Group-K-Machine-Learning/
│
├── README.md                        ← You are here
│
├── Final_Sentiment_Analysis_Workflow.ipynb ← Comprehensive modeling & evaluation
│
├── data/
│   ├── raw/
│   │   ├── sentimentdataset.csv          ← Kaggle: Social Media Sentiment Dataset
│   │   ├── Tweets.csv                    ← Kaggle: Twitter US Airline Sentiment
│   │   ├── twitterdataset.csv            ← Kaggle: Twitter Entity Sentiment Analysis
│   │   ├── gmail_raw.csv                 ← Real-world: Exported Student Gmail Data
│   │   └── whatsapp_raw.csv              ← Real-world: Exported Student WhatsApp Data
│   └── processed/
│       ├── processed_training_dataset.csv   ← 70% stratified training split
│       ├── processed_validation_datset.csv  ← 30% stratified validation split
│       └── student_test_dataset.csv         ← Curated cross-domain test set
│
├── notebooks/
│   ├── 00_data_cleaning.ipynb              ← Cleaning & balancing workflow
│   └── 05_bert_distillbert.ipynb           ← Transformer Fine-Tuning & SHAP
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

After cleaning, label normalisation, deduplication, and balancing, the final combined dataset is split into `processed_training_dataset.csv` and `processed_validation_datset.csv`.

### Custom Test Set (Real-World Evaluation)

To evaluate cross-domain generalization, we manually assembled a unique test set from sources outside the training distribution.

| Source | Description | Size |
|---|---|---|
| **Gmail Exports** | Professional & academic correspondence (Scholarships, Admissions, Newsletters) | 300 samples |
| **WhatsApp Chats** | Informal everyday communication (Social groups, personal messages) | 300 samples |

---

## Methodology

```
Raw Data (3 datasets)
        ↓
Data Cleaning & EDA             [notebooks/00_data_cleaning.ipynb]
• Load 3 raw datasets
• Standardise to text + sentiment schema
• Normalise sentiment labels to Positive/Neutral/Negative
• Remove duplicates and balance classes
        ↓
Combined balanced dataset
        ↓
Stratified split (70/30)
        ↓
processed_training_dataset.csv + processed_validation_datset.csv
```

---

## Algorithms

We implement ten algorithms spanning three generations of NLP approaches.

### Classical Machine Learning

| Algorithm | Trained By | Key Characteristic |
|---|---|---|
| **Naive Bayes** | Larry | Probabilistic; fast; strong baseline |
| **Logistic Regression** | Ritah | Linear; interpretable; efficient |
| **Support Vector Machine (SVM)** | Ivy | Finds optimal decision boundary |
| **Random Forest** | Julianah | Ensemble of decision trees |

### Deep Learning

| Algorithm | Trained By | Key Characteristic |
|---|---|---|
| **TextCNN** | Larry | Captures local n-gram patterns |
| **LSTM** | Ritah | Handles long-range dependencies |
| **Bidirectional LSTM** | Ivy | Dual-direction context modeling |
| **GRU** | Julianah | Efficient recurrent modeling |

### Transformer Models (Pre-trained)

These models leverage **transfer learning** and represent modern transformer-based architectures.

| Algorithm | Trained By | Key Characteristic |
|---|---|---|
| **BERT** | David | Bidirectional transformer using self-attention |
| **DistilBERT** | David | Compressed and efficient version of BERT |

### Ensemble

The final notebook combines models using **weighted voting** based on validation performance.

---

## Evaluation Strategy

Each model is evaluated on two separate test sets:

**1. In-domain test split** — a held-out portion of the training data.
**2. Cross-domain test set** — our custom test set from Gmail and WhatsApp.

Metrics reported for every model: Accuracy, Precision, Recall, and F1 Score.

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
pandas, numpy, scikit-learn, matplotlib, seaborn, tensorflow, transformers, torch
```

Install all dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn tensorflow transformers torch
```

---

## How to Run

**Step 1 — Clone the repository**

```bash
git clone https://github.com/your-username/Group-K-Machine-Learning.git
cd Group-K-Machine-Learning
```

**Step 2 — Run data cleaning**

Run `notebooks/00_data_cleaning.ipynb` to produce the processed datasets.

**Step 3 — Run modeling and evaluation**

Run `Final_Sentiment_Analysis_Workflow.ipynb` to train models and evaluate performance.

---

*Group K — Makerere University*
*Machine Learning Course — 2026*
