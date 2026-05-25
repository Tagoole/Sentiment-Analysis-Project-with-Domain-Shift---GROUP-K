# Final Project Guide: 10-Model Stacking Ensemble for Real-World Sentiment Analysis

## 🎯 Project Overview
This project solves the **Domain Shift** problem in Sentiment Analysis. Most models are trained on social media (Twitter), but struggle when applied to **Real-World Student Communications** (Gmail, WhatsApp, and Technical Alerts). 

To solve this, we have built a **10-Model Stacking Ensemble**. Instead of relying on one algorithm, we combine the strengths of 10 different models using an **XGBoost Meta-Learner**.

---

## 🏗️ The 4-Tier Engineering Strategy

### Tier 1: "Surgical" Data Cleaning (Handling the Noise)
Real-world data is messy. Our pipeline includes:
- **Metadata Extraction:** Before cleaning, we capture signals like `is_all_caps`, `exclamation_count`, and `has_service_alert` (e.g., "Invoice", "Billing").
- **Surgical Cleaner:** Removes technical noise (MTN prefixes, GitHub headers, automated footers) that normally confuses AI models.

### Tier 2: The 10 Base Models (The Team)
We use a diverse "board of directors" to analyze each message:
1.  **Naive Bayes:** Probabilistic speed for keyword detection.
2.  **Logistic Regression:** Identifying word-importance through TF-IDF.
3.  **SVM:** High-dimensional boundary detection.
4.  **Random Forest:** Non-linear decision trees.
5.  **MLP (Multi-Layer Perceptron):** A dense neural network for complex feature mapping.
6.  **TextCNN:** Deep learning that finds "local phrases" (motifs).
7.  **LSTM-Attention:** Focuses on the most emotional parts of a sentence.
8.  **Bidirectional LSTM:** Understands context from both directions.
9.  **Bidirectional GRU:** A faster, gated recurrent model.
10. **Domain-Adapted DistilBERT:** State-of-the-art Transformer that combines deep language understanding with our custom "Real-World" metadata.

### Tier 3: The Stacking Feature Matrix
We don't just take the "Positive/Negative" answer. We take the **Probability** (e.g., "Model A is 80% sure it's Positive"). We stack these probabilities from all 10 models to create a new "Expert Feature Matrix."

### Tier 4: The Meta-Learner (XGBoost)
The 10th model isn't just a member; it's the **Manager**. **XGBoost** looks at the predictions of the 10 base models and learns who is most trustworthy for different types of student messages.

---

## 📊 Why This Approach Wins
- **Generalization:** By combining 10 models, we "average out" the errors of individual algorithms.
- **Domain Awareness:** The metadata features tell the AI: *"This is a technical alert, don't mistake 'failed build' for a personal negative emotion."*
- **Collaboration:** This architecture allows every team member's specific model (from their individual notebooks) to contribute to the final high accuracy.

---

## 🚀 Kaggle Instructions
1.  **Accelerator:** Use **GPU P100** or **T4**.
2.  **Files:** Upload `processed_training_dataset.csv`, `processed_validation_datset.csv`, and `student_test_dataset.csv`.
3.  **Run Order:** Run all cells in order. The notebook is designed to train each model explicitly before moving to the Stacking phase.
