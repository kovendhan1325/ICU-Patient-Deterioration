import pandas as pd
import os

def markdown_to_excel(md_path, excel_path):
    with open(md_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    in_table = False
    table_lines = []
    
    for line in lines:
        if line.strip().startswith("| # | Feature |"):
            in_table = True
            
        if in_table:
            if line.strip() == "":
                break
            table_lines.append(line.strip())
            
    # Parse table manually
    headers = [col.strip() for col in table_lines[0].split('|') if col.strip() != '']
    
    data = []
    # Skip header and separator (lines 0 and 1)
    for line in table_lines[2:]:
        cols = [col.strip().strip('`') for col in line.split('|') if col.strip() != '']
        if len(cols) == len(headers):
            data.append(cols)
            
    df = pd.DataFrame(data, columns=headers)
    
    # Save to Excel
    df.to_excel(excel_path, index=False)
    print(f"Successfully saved {len(df)} features to {excel_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    md_path = os.path.join(base_dir, "preprocessing_pipeline", "reports", "data_dictionary.md")
    excel_path = os.path.join(base_dir, "lstm_features.xlsx")
    
    markdown_to_excel(md_path, excel_path)
