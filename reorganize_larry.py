import json

def fix_larry_nb_dependency(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    new_cells = []
    vectorization_code = None
    tuning_cell = None
    final_training_cell = None
    
    # 1. Extract the pieces we need to reorganize
    for cell in nb['cells']:
        source = "".join(cell['source'])
        
        # This cell currently contains both vectorization and final training
        if "# 1. Vectorization with TF-IDF" in source and "nb_clf = MultinomialNB" in source:
            # We split this cell into two: Vectorization and Final Training
            lines = cell['source']
            vec_lines = []
            train_lines = []
            is_train = False
            for line in lines:
                if "# 2. Initialize and Train Naive Bayes with BEST alpha" in line:
                    is_train = True
                if is_train:
                    train_lines.append(line)
                else:
                    vec_lines.append(line)
            
            # Create a separate Vectorization cell
            vectorization_code = {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": vec_lines
            }
            # Create a separate Final Training cell
            final_training_cell = {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": train_lines
            }
            continue # Don't add the original combined cell
            
        # This is the tuning loop
        if "for a in [0.01, 0.05, 0.1, 0.5, 1.0]:" in source:
            tuning_cell = cell
            continue # Don't add it yet
            
        new_cells.append(cell)

    # 2. Re-insert them in the correct logical order
    final_cells = []
    for cell in new_cells:
        source = "".join(cell['source'])
        final_cells.append(cell)
        
        # Insert our reorganized cells after the Step 6 header
        if "### Step 6: Model 1 - Naive Bayes" in source:
            if vectorization_code:
                final_cells.append(vectorization_code)
            if tuning_cell:
                final_cells.append(tuning_cell)
            if final_training_cell:
                final_cells.append(final_training_cell)

    nb['cells'] = final_cells
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Success: Larry's Naive Bayes dependencies fixed.")

if __name__ == "__main__":
    fix_larry_nb_dependency('notebooks/LARRY_NB_TextCNN.ipynb')
