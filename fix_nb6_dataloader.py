"""
Robust fix for Notebook_6 - replaces specific broken lines individually.
No block-matching — finds exact strings line by line.
"""
import json, os

NB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'Notebook_6_Improved_Models.ipynb')

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

REPLACEMENTS = {
    # Fix 1: remove verbose=True from ReduceLROnPlateau
    "scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)\n":
    "scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)\n",

    # Fix 2: safe pos_weight computation (scalar tensor is fine for BCEWithLogitsLoss)
    "pos_weight = (len(ye_train) - ye_train.sum()) / max(ye_train.sum(), 1)\n":
    "pos_weight = float(len(ye_train) - int(ye_train.sum())) / float(max(int(ye_train.sum()), 1))\n",

    # Fix 3: scheduler.step manual LR logging in training loop
    "    scheduler.step(val_auc)\n":
    "    _prev_lr = optimizer.param_groups[0]['lr']\n"
    "    scheduler.step(val_auc)\n"
    "    if optimizer.param_groups[0]['lr'] < _prev_lr:\n"
    "        print(f'  [LR] Reduced to {optimizer.param_groups[0][\"lr\"]:.6f}')\n",
}

total_fixed = 0
for cell in nb.get('cells', []):
    if cell.get('cell_type') != 'code':
        continue
    new_src = []
    for line in cell.get('source', []):
        if line in REPLACEMENTS:
            new_src.append(REPLACEMENTS[line])
            total_fixed += 1
            print(f'  FIXED: {line.strip()[:70]}')
        else:
            new_src.append(line)
    cell['source'] = new_src

print(f'\nTotal lines fixed: {total_fixed}')

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print('Notebook saved.')
