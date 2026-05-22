
---

## 🛑 THE MANDATORY FIX: The Three-Class Rule (Everyone)

I checked the processed test files—over **30% of our test data is Neutral** (alerts, receipts, invoices). 

**The Problem:** If your notebook only maps "Positive" and "Negative," every single Neutral email in our test set is being deleted or ignored by your model. That’s why our accuracy is tanking!
**The Action:** You MUST update your label mapping to handle **3 labels**:
```python
# Ensure this is in your notebook!
df['label'] = df['sentiment'].map({"Negative": 0, "Neutral": 1, "Positive": 2})
```
Update your `num_labels=3` (for David) or your final `Dense(3, activation='softmax')` (for Larry, Ritah, Ivy, Julianah).

---

## 🛠️ Stage 2: Surgical Cleaning (Add this to your Preprocessing)

In addition to what I've already cleaned, please ensure your notebook-specific cleaning functions handle these 4 things. This is what separates "student" models from "industry" models.

### 1. Technical Noise Removal (Critical for Gmail/WhatsApp)
Our test data is full of automated tech noise. Even if I've removed the big chunks, your model might still see "MTN:", "GitHub:", or "Forwarded message".
*   **Action:** Add a regex line to strip out service prefixes and email footers like *"Please consider the environment"*. If the model sees "Environment" in every neutral email, it will think that word means neutral, which is wrong.

### 2. Case Normalization & Contractions
*   **Action:** Ensure you are converting everything to `.lower()`. Also, expand contractions (e.g., convert "don't" to "do not"). This helps classical models like **LogReg (Ritah)** and **SVM (Ivy)** realize that "Don't" and "do not" are the same sentiment.

### 3. Slang & Shortcut Mapping (The "Student Life" Fix)
Since we're keeping everything in English, we need to bridge the gap between "WhatsApp English" and "Formal English."
*   **Action:** If your model (like **Naive Bayes - Larry**) sees "assignment is due tmrw" it might get confused. Ensure your cleaning function maps common shortcuts: `"u": "you"`, `"pls": "please"`, `"tmrw": "tomorrow"`, `"wat": "what"`.

### 4. Special Character Handling (The "Shouting" Factor)
*   **Action:** In student chats, sentiment is often in the punctuation (e.g., "I failed!!!" vs "I failed").
    *   **For David (BERT):** Leave the punctuation in; BERT understands it.
    *   **For Julianah (Random Forest):** Create a new feature called `exclamation_count` before you strip the punctuation. It’s a huge clue for sentiment!

---

## 💡 Algorithm-Specific "Pro" Tips

### David (BERT & DistilBERT)
*   **Context Window:** Increase `max_length` to **512**. 128 is too short; you're cutting off the actual sentiment of the emails.
*   **Domain Training:** Use the raw `gmail_raw.csv` I provided to do 3 epochs of "Masked Language Modeling" before you fine-tune. It teaches BERT our vocabulary.

### Larry (Naive Bayes & TextCNN)
*   **TextCNN Filters:** Use filters of `[3, 5, 7, 9]`. This helps the CNN see the "Service Alert" (short) and the "Email Body" (long) at the same time.
*   **NB N-Grams:** Use `ngram_range=(1, 3)`. Single words aren't enough for emails.

### Ritah (Logistic Regression & LSTM)
*   **LogReg Regularization:** Use `penalty='l1'`. It will automatically ignore the "noise" words I missed in the first cleaning pass.
*   **LSTM Attention:** Add an Attention Layer. It helps the model ignore the "Environment" footer and focus on the "Approved" or "Rejected" part of the email.

### Ivy (SVM & Bi-LSTM)
*   **SVM Kernel:** Use `kernel='rbf'`. Linear lines can't separate messy WhatsApp data.
*   **Bi-LSTM Dropout:** Use **SpatialDropout1D**. It’s the industry standard for making Bi-LSTMs more robust to different writing styles.

### Julianah (Random Forest & GRU)
*   **RF Metadata:** Since I've cleaned the text, you should create features like `has_html_artifacts` or `is_all_caps`. RF loves these structural hints.
*   **GRU Clipping:** Use `clipnorm=1.0` to stop long sentences from breaking your gradients.

---
