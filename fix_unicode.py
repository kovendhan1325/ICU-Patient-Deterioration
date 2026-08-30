import sys, io
# Force UTF-8 output on Windows — prevents UnicodeEncodeError on cp1252 terminals
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

with open('train_bilstm_continue.py', 'r', encoding='utf-8') as f:
    content = f.read()

replacements = {
    '\u2192': '->',
    '\u2190': '<-',
    '\u2714': '[OK]',
    '\u2718': '[X]',
    '\u2588': '#',
    '\u2713': '[OK]',
    '\u2717': '[X]',
    '\u2500': '-',
    '\u2502': '|',
}
for uni, asc in replacements.items():
    content = content.replace(uni, asc)

# Prepend UTF-8 stdout fix at top of the training script
header = '# -*- coding: utf-8 -*-\nimport sys, io\nsys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")\n'
if 'TextIOWrapper' not in content:
    content = header + content

with open('train_bilstm_continue.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Fixed all unicode characters in train_bilstm_continue.py')
