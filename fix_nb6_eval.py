import json, os

NB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Notebook_6_Improved_Models.ipynb')

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell.get('cell_type') != 'code':
        continue
    new_src = []
    for line in cell.get('source', []):
        if "model.load_state_dict(torch.load(LSTM_MODEL_PATH))" in line:
            new_src.append(line.replace("model.load_state_dict(torch.load(LSTM_MODEL_PATH))", "model.load_state_dict(torch.load(LSTM_MODEL_PATH, map_location=device))"))
        else:
            new_src.append(line)
    cell['source'] = new_src

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print('Notebook updated with map_location=device.')
