import json

def fix_ivy(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    for cell in nb['cells']:
        source = "".join(cell['source'])
        if "I'll train for 5 epochs with our best learning rate of" in source:
            # Fix the broken Python concatenation in JSON string
            cell['source'] = [
                "### Step 8: Training the Final Bi-LSTM\n",
                "I'll train for 5 epochs with our best learning rate found in the experiment above."
            ]
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

if __name__ == "__main__":
    fix_ivy('notebooks/IVY_SVM_BiLSTM.ipynb')
