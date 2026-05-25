
import json
import re

def fix_notebook(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Identify the corrupted part. 
    # It starts after the "Surgical cleaning complete!" cell ends.
    # We'll search for the signature of the corruption.
    
    corrupt_start_pattern = r'\"Surgical cleaning complete!\"\)\"\s*\]\s*\}\s*,\s*\{'
    corrupt_end_pattern = r'\"Surgical cleaning is done!\"\s*\]\s*\}'
    
    # We need to replace the entire block from the start of the corrupt cell to its end.
    # The corruption is basically a cell that is missing its keys.
    
    # Let's find the indices
    start_match = re.search(corrupt_start_pattern, content)
    if not start_match:
        print("Could not find start of corruption.")
        return
    
    # The corruption starts after the first } after "Surgical cleaning complete!"
    # Wait, the content I read showed:
    # "print(\"Surgical cleaning complete!\")"\n   ]\n  },\n  {
    # So it's the second { that starts the corruption.
    
    end_match = re.search(corrupt_end_pattern, content)
    if not end_match:
        print("Could not find end of corruption.")
        return

    # Reconstruct the "surgical_cleaner" cell properly.
    # I'll combine the good parts of both.
    
    new_cell_source = [
        "import re\n",
        "import string\n",
        "\n",
        "slang_dictionary = {\n",
        "    r'\\bu\\b': 'you', r'\\bpls\\b': 'please', r'\\bplz\\b': 'please',\n",
        "    r'\\btmrw\\b': 'tomorrow', r'\\bwat\\b': 'what', r'\\bwud\\b': 'would',\n",
        "    r'\\bcuz\\b': 'because', r'\\bbtw\\b': 'by the way', r'\\bidk\\b': 'i do not know',\n",
        "    r'\\bomg\\b': 'oh my god', r'\\blol\\b': 'laughing', r'\\bthx\\b': 'thanks',\n",
        "    r'\\bsry\\b': 'sorry', r'\\bwanna\\b': 'want to', r'\\bgonna\\b': 'going to',\n",
        "    r'\\bur\\b': 'your', r'\\br\\b': 'are', r'\\bn\\b': 'and', r'\\bok\\b': 'okay'\n",
        "}\n",
        "\n",
        "def extract_meta_features(df):\n",
        "    df['exclamation_count'] = df['text'].apply(lambda x: str(x).count('!'))\n",
        "    df['is_all_caps'] = df['text'].apply(lambda x: 1 if str(x).isupper() and len(str(x)) > 5 else 0)\n",
        "    df['char_cnt'] = df['text'].apply(lambda x: len(str(x)))\n",
        "    df['word_cnt'] = df['text'].apply(lambda x: len(str(x).split()))\n",
        "    \n",
        "    # Neutral Trap Fix: Platform Mentions\n",
        "    df['has_platform_mention'] = df['text'].apply(\n",
        "        lambda x: 1 if re.search(r'\\b(github|slack|coursera|udemy|paystack|railway|netlify|heroku|airtel|mtn|vultr|google|microsoft|gmail|whatsapp)\\b', str(x).lower()) else 0\n",
        "    )\n",
        "    # Neutral Trap Fix: Service Alerts\n",
        "    df['has_service_alert'] = df['text'].apply(\n",
        "        lambda x: 1 if re.search(r'\\b(invoice|billing|terminate|security|alert|reminder|otp|verification|payment|warning)\\b', str(x).lower()) else 0\n",
        "    )\n",
        "    return df\n",
        "\n",
        "def surgical_cleaner(text):\n",
        "    if not isinstance(text, str): return \"notification\"\n",
        "    text = text.lower()\n",
        "    \n",
        "    # Remove technical noise and prefixes\n",
        "    text = re.sub(r'http\\S+|www\\.\\S+|@\\w+', '', text)\n",
        "    text = re.sub(r'<.*?>', '', text)\n",
        "    \n",
        "    service_prefixes = (\n",
        "        r'^(mtn|airtel|github|slack|gmail|whatsapp|udemy|vultr|codemagic|linkedin|' \n",
        "        r'amazon|railway|netlify|heroku|paystack|digitalocean|vercel|coursera|' \n",
        "        r'service termination notice|billing alert|security alert|reminder|alert|' \n",
        "        r'forwarded message|from:|to:|subject:|date:|sent:|on behalf of):\\s*'\n",
        "    )\n",
        "    text = re.sub(service_prefixes, '', text)\n",
        "    \n",
        "    # Remove signatures\n",
        "    text = re.sub(r'sent from my (iphone|android|mobile)', '', text)\n",
        "    text = re.sub(r'please consider the environment before printing', '', text)\n",
        "    \n",
        "    for slang, correct in slang_dictionary.items():\n",
        "        text = re.sub(slang, correct, text)\n",
        "        \n",
        "    text = text.translate(str.maketrans('', '', string.punctuation))\n",
        "    text = re.sub(r'\\s+', ' ', text).strip()\n",
        "    return text if text else \"notification\"\n",
        "\n",
        "print(\"Extracting features...\")\n",
        "train_df = extract_meta_features(train_df)\n",
        "val_df   = extract_meta_features(val_df)\n",
        "test_df  = extract_meta_features(test_df)\n",
        "\n",
        "print(\"Cleaning text...\")\n",
        "train_df['clean_text'] = train_df['text'].apply(surgical_cleaner)\n",
        "val_df['clean_text']   = val_df['text'].apply(surgical_cleaner)\n",
        "test_df['clean_text']  = test_df['text'].apply(surgical_cleaner)\n",
        "print(\"Done!\")\n"
    ]
    
    # We replace the corrupt block with a proper cell
    new_cell = {
        "cell_type": "code",
        "execution_count": None,
        "id": "fixed_surgical_cleaner",
        "metadata": {},
        "outputs": [],
        "source": new_cell_source
    }
    
    # Since I can't easily do string replacement on malformed JSON and then parse,
    # I'll try to manually fix the string so it's parsable.
    
    # Find the start of the corrupt cell's opening brace
    corrupt_brace_start = content.find('{', start_match.end() - 5) 
    # Wait, the start_match.end() is after " {".
    
    # Actually, let's just use the indices to slice and dice.
    # The corruption is between line 254 and line 319 (approx).
    
    # I'll use a more robust approach: find the "Surgical cleaning complete!" cell and replace it and the following mess.
    
    # Let's find the "cell_type": "code" that contains "Surgical cleaning complete!"
    # And replace it and the following corrupted block with my new cell.
    
    # I'll just load the whole file as a list of lines, and find the lines to replace.
    lines = content.splitlines()
    start_line = -1
    end_line = -1
    for i, line in enumerate(lines):
        if "X0SCkEW82IAH" in line: # ID of the cell before/at corruption
            start_line = i - 5 # Back up to the start of the cell
        if "Surgical cleaning is done!" in line:
            end_line = i + 2 # End of the corrupted block
            break
            
    if start_line != -1 and end_line != -1:
        new_lines = lines[:start_line] + [json.dumps(new_cell, indent=1)] + lines[end_line+1:]
        # Wait, I need to make sure the comma is correct.
        fixed_content = "\n".join(lines[:start_line-1]) + ",\n" + json.dumps(new_cell, indent=1) + ",\n" + "\n".join(lines[end_line+1:])
    else:
        print("Manual line search failed. Trying regex.")
        # Fallback to regex replacement if IDs changed
        # ...
        return

    # Now parse the fixed content to JSON to apply the rest of the changes
    try:
        nb = json.loads(fixed_content)
    except Exception as e:
        print(f"JSON parsing failed after fix: {e}")
        # Let's write the "fixed" but still potentially broken content to a temp file for debugging
        with open('debug_fixed.ipynb', 'w') as df:
            df.write(fixed_content)
        return

    # --- Apply Architectural Changes ---
    
    # 1. Update MAX_LEN
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            if "MAX_LEN    = 120" in source:
                cell['source'] = [s.replace("MAX_LEN    = 120", "MAX_LEN    = 150") for s in cell['source']]
    
    # 2. Add Meta Scaling and Functional API Model
    
    meta_cols = ['exclamation_count', 'is_all_caps', 'char_cnt', 'word_cnt', 'has_platform_mention', 'has_service_alert']
    
    # Find the model build cell
    for cell in nb['cells']:
        if cell['cell_type'] == 'code' and 'def build_lstm_glove' in "".join(cell['source']):
            cell['source'] = [
                "from tensorflow.keras.layers import Bidirectional, GRU, GlobalAveragePooling1D, concatenate\n",
                "\n",
                "def build_functional_gru(num_words, embed_dim, embedding_matrix, max_len, meta_dim, num_classes, lr=3e-4):\n",
                "    # Text Branch\n",
                "    text_input = Input(shape=(max_len,), name='text_input')\n",
                "    emb = Embedding(num_words, embed_dim, weights=[embedding_matrix], trainable=True)(text_input)\n",
                "    x = SpatialDropout1D(0.3)(emb)\n",
                "    x = Bidirectional(GRU(64, return_sequences=True))(x)\n",
                "    avg_pool = GlobalAveragePooling1D()(x)\n",
                "    max_pool = GlobalMaxPooling1D()(x)\n",
                "    text_branch = concatenate([avg_pool, max_pool])\n",
                "    \n",
                "    # Meta Branch\n",
                "    meta_input = Input(shape=(meta_dim,), name='meta_input')\n",
                "    m = Dense(16, activation='relu')(meta_input)\n",
                "    \n",
                "    # Combined\n",
                "    combined = concatenate([text_branch, m])\n",
                "    combined = Dropout(0.5)(combined)\n",
                "    combined = Dense(64, activation='relu')(combined)\n",
                "    output = Dense(num_classes, activation='softmax')(combined)\n",
                "    \n",
                "    model = Model(inputs=[text_input, meta_input], outputs=output)\n",
                "    model.compile(optimizer=Adam(learning_rate=lr, clipnorm=1.0), \n",
                "                  loss='categorical_crossentropy', metrics=['accuracy'])\n",
                "    return model\n",
                "\n",
                "print('Functional GRU Model ready!')\n"
            ]
        
        # Update preprocessing to include scaler
        if cell['cell_type'] == 'code' and 'X_train_pad =' in "".join(cell['source']):
            cell['source'].append("\n# Scaling meta features\n")
            cell['source'].append(f"meta_cols = {meta_cols}\n")
            cell['source'].append("scaler = StandardScaler()\n")
            cell['source'].append("X_train_meta = scaler.fit_transform(train_df[meta_cols])\n")
            cell['source'].append("X_val_meta   = scaler.transform(val_df[meta_cols])\n")
            cell['source'].append("X_test_meta  = scaler.transform(test_df[meta_cols])\n")

    # Update training cell to use the new model and inputs
    for cell in nb['cells']:
        if cell['cell_type'] == 'code' and 'for lr_val in [1e-3, 3e-4]:' in "".join(cell['source']):
            cell['source'] = [
                "best_lr_gru = 3e-4\n",
                "best_val_acc_gru = 0\n",
                "meta_dim = X_train_meta.shape[1]\n",
                "\n",
                "for lr_val in [1e-3, 3e-4]:\n",
                "    print(f'\\n--- Testing GRU with Learning Rate: {lr_val} ---')\n",
                "    temp_model = build_functional_gru(num_words, EMBED_DIM, embedding_matrix, MAX_LEN, meta_dim, NUM_CLASSES, lr=lr_val)\n",
                "    \n",
                "    temp_hist = temp_model.fit(\n",
                "        {'text_input': X_train_pad, 'meta_input': X_train_meta}, \n",
                "        y_train_cat, \n",
                "        validation_data=({'text_input': X_val_pad, 'meta_input': X_val_meta}, y_val_cat), \n",
                "        epochs=2, batch_size=64, verbose=1\n",
                "    )\n",
                "    \n",
                "    val_acc = max(temp_hist.history['val_accuracy'])\n",
                "    if val_acc > best_val_acc_gru:\n",
                "        best_val_acc_gru = val_acc\n",
                "        best_lr_gru = lr_val\n",
                "\n",
                "print(f'\\nBest GRU Learning Rate: {best_lr_gru}')\n"
            ]

    # Save fixed notebook
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Notebook fixed and updated successfully.")

fix_notebook('notebooks/LOGISTIC REGRESSION AND LSTM-RITAH.ipynb')
