# Machine Learning Workflow & Marking Guide
## Project: Evaluating Sentiment Model Generalization: From Social Media to Student Communications

This document outlines the structural roadmap and workflow implemented in this project, focusing on technical clarity and evaluation on real-world data.

---

## 1. Problem Definition & Introduction
### The Problem
Traditional sentiment analysis models are often trained on generic data and may struggle when applied to specific domains like student life. Students use unique vocabulary and technical artifacts in formal (Gmail) and informal (WhatsApp) communications that standard models might misinterpret.
### Objectives of the project
*   **Domain Adaptation:** Adapt models to the academic context by fine-tuning on student-specific vocabulary.
*   **Performance Comparison:** Evaluate the trade-offs between classical (SVM/NB) and transformer-based (BERT/DistilBERT) models.
*   **Interpretability:** Use SHAP and LIME to explain model decisions and ensure clarity.

---

## 2. Data Collection & Acquisition
*   **Sources:** Integration of multiple datasets including Twitter and real-world student data from Gmail and WhatsApp.
*   **Structure:** Aggregation of raw CSV files into a unified repository structure.

---

## 3. Exploratory Data Analysis (EDA)
*   **Distribution Analysis:** Visualization of sentiment class balance.
*   **Source Contribution:** Tracking which data sources provide which sentiments.
*   **Text Length Analysis:** Identifying the difference in sentence length between formal emails and informal chats.

---

## 4. Data Cleaning & Preprocessing
*   **Data Cleaning (Generic):** HTML stripping, case normalization, and whitespace cleanup.
*   **Algorithm-Specific Preprocessing:**
    *   **BERT/Deep Learning:** Preservation of punctuation and sentence structure.
    *   **Classical Models (SVM/NB):** Stop-word removal and lemmatization.
    *   **Slang Mapping:** Converting informal abbreviations to standard text to normalize the vocabulary.

---

## 5. Feature Engineering
*   **Word Embeddings (Deep Learning):** Tokenization and embedding generation using modern libraries.
*   **Vectorization (Classical):** TF-IDF and N-Gram (1-3) generation for Naive Bayes and SVM.

---

## 6. Model Training & Implementation
Each model is implemented in its own dedicated section to ensure clarity:
*   **David (DistilBERT):** Transformer-based modeling with domain adaptation.
*   **Larry (Naive Bayes & TextCNN):** Baseline probabilistic modeling vs. Convolutional Neural Networks.
*   **Ritah (Logistic Regression & LSTM):** Linear classification vs. Recurrent Neural Networks.
*   **Ivy (SVM & Bi-LSTM):** High-dimensional kernel separation vs. Bidirectional sequence modeling.
*   **Julianah (Random Forest & GRU):** Ensemble decision trees vs. Gated Recurrent Units.

---

## 7. Model Evaluation (Individual & Comparative)
### Individual Evaluation
Each model reports:
*   **Precision, Recall, and F1-Score** (Macro-averaged).
*   **Confusion Matrix** to identify where the model makes errors.
### Comparative Evaluation
*   **Performance Comparison:** A table comparing all models on Accuracy and F1-Score.
*   **Discussion:** Analysis of performance trade-offs between different architectures.

---

## 8. Model Interpretability (SHAP & LIME)
*   **Global Analysis:** Examining the weights of the classification head.
*   **Local Attribution (SHAP):** Visualizing word-level contributions to the model's prediction.

---

## 9. Domain Evaluation (Real-World Inference)
*   **Scenario Testing:** Passing raw, unseen student narratives (Gmail/WhatsApp) through the final system.
*   **Verification:** Assessing the model's verdict and confidence levels on new data.

---

## 10. Conclusion
*   Summary of findings.
*   Recommendations based on model performance.
