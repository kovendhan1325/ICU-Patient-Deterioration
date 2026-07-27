import json
import glob

for nb_file in glob.glob("Notebook_*.ipynb"):
    with open(nb_file, "r", encoding="utf-8") as f:
        nb = json.load(f)
    
    modified = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            source = cell.get("source", [])
            new_source = []
            for line in source:
                if "from google.colab import drive" in line:
                    modified = True
                    continue
                if "drive.mount(" in line:
                    modified = True
                    continue
                if "data_dir =" in line and "/content/drive/" in line:
                    new_source.append("data_dir = 'preprocessing_pipeline/output'\n")
                    modified = True
                else:
                    new_source.append(line)
            
            # if we removed lines, we might have consecutive newlines at the start, but that's fine.
            if len(source) != len(new_source) or source != new_source:
                cell["source"] = new_source
                modified = True
                
    if modified:
        with open(nb_file, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=2)
        print(f"Updated {nb_file}")
