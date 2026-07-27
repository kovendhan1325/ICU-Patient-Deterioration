import json
import glob

def make_paths_dynamic():
    for nb_file in glob.glob("Notebook_*.ipynb"):
        with open(nb_file, "r", encoding="utf-8") as f:
            nb = json.load(f)
        
        modified = False
        for cell in nb.get("cells", []):
            if cell.get("cell_type") == "code":
                source = "".join(cell.get("source", []))
                
                # Check if this is the cell that defines data_dir
                if "data_dir = 'preprocessing_pipeline/output'" in source and "try:" not in source:
                    new_source = [
                        "import os\n",
                        "try:\n",
                        "    from google.colab import drive\n",
                        "    # Only mount if the drive isn't already mounted to avoid errors\n",
                        "    if not os.path.exists('/content/drive/MyDrive'):\n",
                        "        drive.mount('/content/drive')\n",
                        "    data_dir = '/content/drive/MyDrive/ICU-Patient-Deterioration/preprocessing_pipeline/output'\n",
                        "    print('Running in Google Colab. Data directory set to Google Drive.')\n",
                        "except ImportError:\n",
                        "    # If not in Colab (running locally)\n",
                        "    data_dir = 'preprocessing_pipeline/output'\n",
                        "    print('Running Locally. Data directory set to local folder.')\n"
                    ]
                    cell["source"] = new_source
                    modified = True
                    
        if modified:
            with open(nb_file, "w", encoding="utf-8") as f:
                json.dump(nb, f, indent=2)
            print(f"Updated {nb_file} to be dynamic (Colab + Local)")

if __name__ == "__main__":
    make_paths_dynamic()
