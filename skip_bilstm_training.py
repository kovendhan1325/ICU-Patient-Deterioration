import json, os

NB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Notebook_6_Improved_Models.ipynb')

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell.get('cell_type') != 'code':
        continue
    
    # Check for the training loop cell
    if any("for epoch in range(EPOCHS):" in line for line in cell.get('source', [])):
        # Add logic to skip training if the model exists
        new_src = []
        new_src.append("import os\n")
        new_src.append("LSTM_MODEL_PATH = os.path.join(BASE_DIR, 'best_bilstm_model.pth')\n")
        new_src.append("if os.path.exists(LSTM_MODEL_PATH):\n")
        new_src.append("    print(f'Found existing BiLSTM model at {LSTM_MODEL_PATH}, skipping training.')\n")
        new_src.append("    model.load_state_dict(torch.load(LSTM_MODEL_PATH, map_location=device))\n")
        new_src.append("else:\n")
        
        # Indent the original source
        for line in cell['source']:
            if line == "LSTM_MODEL_PATH  = os.path.join(BASE_DIR, 'best_bilstm_model.pth')\n":
                continue # Already defined above
            new_src.append("    " + line)
            
        cell['source'] = new_src
        break

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print('Patched Notebook 6 to skip BiLSTM training if model exists.')
