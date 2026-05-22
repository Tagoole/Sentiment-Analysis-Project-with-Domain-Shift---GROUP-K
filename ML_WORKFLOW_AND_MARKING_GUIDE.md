# Machine Learning Workflow & Marking Guide
## Project: Cross-Domain Sentiment Analysis for Student Life Narratives

This document serves as the structural roadmap and marking guide for the machine learning workflow implemented in this project. It outlines the technical rigor and pedagogical steps required to build a robust sentiment analysis system.

---

## 1. Problem Definition & Introduction
### The Problem
Traditional sentiment analysis models are often trained on generic data (like movie reviews) and fail when applied to niche, high-stakes domains like student life. Students use unique vocabulary, slang, and technical artifacts (from Gmail/WhatsApp) that standard models misinterpret.
### Objectives
*   **Domain Adaptation:** Bridge the gap between generic English and "Student English."
*   **Class Imbalance Resolution:** Handle the high frequency of Neutral/Informational texts.
*   **Model Comparison:** Evaluate the trade-offs between classical (SVM/NB) and deep learning (BERT/LSTM) approaches.
*   **Interpretability:** Use SHAP to explain *why* the model makes specific decisions.

---

## 2. Data Collection & Acquisition
*   **Sources:** Multi-source integration (Tweets, SentimentDataset, Gmail Exports, WhatsApp Chats).
*   **Structure:** Aggregation of raw CSV files into a unified repository structure.
*   **Marking Note:** Look for the variety of sources and the raw-to-processed transition in the `data/` folder.

---

## 3. Exploratory Data Analysis (EDA)
*   **Distribution Analysis:** Visualization of sentiment class balance before and after merging.
*   **Source Contribution:** Tracking which data sources provide which sentiments.
*   **Text Length Analysis:** Identifying the difference in sentence length between formal emails and informal chats.
*   **Marking Note:** Review `notebooks/00_data_cleaning.ipynb` for distribution plots and statistical summaries.

---

## 4. Data Cleaning & Preprocessing
*   **Surgical Cleaning (Generic):** HTML stripping, case normalization, and whitespace cleanup.
*   **Algorithm-Specific Preprocessing:**
    *   **BERT/Deep Learning:** Preservation of punctuation and sentence structure.
    *   **Classical Models (SVM/NB):** Aggressive stop-word removal and lemmatization to reduce noise.
    *   **Student Slang Mapping:** Converting "u" -> "you", "tmrw" -> "tomorrow" to standardize the vocabulary.

---

## 5. Feature Engineering
*   **Word Embeddings (Deep Learning):** Tokenization and dynamic embedding generation (AutoTokenizer).
*   **Vectorization (Classical):** TF-IDF and N-Gram (1-3) generation for Naive Bayes and SVM.
*   **Structural Features:** (Optional for RF/GRU) Tracking exclamation counts and all-caps frequency as sentiment indicators.

---

## 6. Model Training & Implementation
Each model must be implemented in its own dedicated section/cell to ensure clarity:
*   **David (DistilBERT):** Two-stage training (MLM Domain Adaptation + Sequence Classification).
*   **Larry (Naive Bayes & TextCNN):** Baseline probabilistic modeling vs. 1D Convolutional Neural Networks.
*   **Ritah (Logistic Regression & LSTM):** Linear classification vs. Recurrent Neural Networks with Attention.
*   **Ivy (SVM & Bi-LSTM):** High-dimensional kernel separation vs. Bidirectional sequence modeling.
*   **Julianah (Random Forest & GRU):** Ensemble decision trees vs. Gated Recurrent Units.

---

## 7. Model Evaluation (Individual & Comparative)
### Individual Evaluation
Each model must report:
*   **Precision, Recall, and F1-Score** (Macro-averaged for 3 classes).
*   **Confusion Matrix** to identify where the model is "confused" (e.g., mistaking Neutral for Negative).
### Comparative Evaluation
*   **Final Leaderboard:** A table comparing all models on Accuracy and ROC-AUC.
*   **Performance vs. Complexity:** Discussion on whether the 40% speed gain of DistilBERT is worth the slight accuracy trade-off.

---

## 8. Model Interpretability (SHAP & LIME)
*   **Global Analysis:** Examining the weights of the classification head.
*   **Local Attribution (SHAP):** Visualizing word-level contributions (e.g., how the word "rejection" pushes the model toward a Negative verdict).
*   **Marking Note:** This demonstrates "Industry-Grade" analysis beyond just accuracy numbers.

---

## 9. Domain Stress-Test (Real-World Inference)
*   **Scenario Testing:** Passing raw, unseen student narratives through the final system.
*   **Verification:** Manual verification of the model's "Verdict" and "Confidence" levels.

---

## 10. Conclusion & Deployment Readiness
*   Summary of findings.
*   Path to production (ONNX/TorchScript export).
*   Future work recommendations.
