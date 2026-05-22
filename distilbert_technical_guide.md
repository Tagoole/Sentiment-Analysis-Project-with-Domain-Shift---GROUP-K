# Technical Guide: DistilBERT Sentiment Analysis for Student Life

This guide explains the advanced NLP workflow implemented in the `05_bert_distillbert.ipynb` notebook. This project achieves high accuracy by combining industrial cleaning techniques with state-of-the-art transformer modeling.

---

## 1. The Workflow Overview
The project uses a **Two-Stage Transfer Learning** approach:
1.  **Domain Adaptation (MLM):** The model "reads" unlabelled student text (Gmail/WhatsApp) to learn the specific vocabulary.
2.  **Fine-Tuning:** The model is trained to classify text into `Positive`, `Neutral`, or `Negative`.

---

## 2. Surgical Cleaning (The Accuracy Engine)
Standard cleaning (removing nulls) isn't enough for Transformers. We use **Surgical Cleaning** to remove "noise" that distracts the model:
*   **Technical Noise:** Strips HTML tags (`<br>`, `<div>`) and platform artifacts (`@user`, `!!!`).
*   **System Footers:** Removes automated text like "Sent from my iPhone" and "Please consider the environment."
*   **Service Markers:** Removes prefixes like `MTN:`, `GitHub:`, and `Vultr:` which BERT often mistakes for sentiment markers.
*   **Slang Mapping:** Converts student shortcuts (`u` -> `you`, `pls` -> `please`) into formal English that BERT understands.

---

## 3. Domain Adaptation (The Road to 85%)
By default, BERT is trained on Wikipedia. It doesn't know what a "retake" or a "coursework deadline" is in a Ugandan student context.
*   **Masked Language Modeling (MLM):** We hide 15% of the words in your raw data and ask the model to guess them.
*   **The Result:** The model becomes "literate" in your specific domain before it ever sees a sentiment label. This is the key to breaking the 71% accuracy barrier.

---

## 4. Training Strategy
*   **Learning Rate Warmup:** We start training with a very small learning rate to avoid "shocking" the pre-trained weights.
*   **Linear Scheduling:** The learning rate slowly decays, allowing the model to "settle" into the most accurate state.
*   **Memory Optimization:** We use **Gradient Accumulation** to train large models on standard computers (like Colab) without crashing the RAM.

---

## 5. Evaluation
We evaluate the model on two levels:
1.  **In-Domain:** Standard test split.
2.  **Cross-Domain (The Stress Test):** Using your real-world Gmail and WhatsApp data. This proves the model works in the "real world," not just on a curated dataset.
