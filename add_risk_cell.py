import json

notebook_path = "Notebook_2_Train_Random_Forest.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

new_cell_source = [
    "import pandas as pd\n",
    "\n",
    "# Create a DataFrame to view the final predictions and risk levels\n",
    "results_df = pd.DataFrame({\n",
    "    'True_Label': y_val,\n",
    "    'Deterioration_Predicted': y_val_pred,\n",
    "    'Deterioration_Probability': y_val_prob\n",
    "})\n",
    "\n",
    "# Define risk categories based on probability thresholds\n",
    "# You can adjust these thresholds based on clinical requirements\n",
    "def categorize_risk(prob):\n",
    "    if prob < 0.25:\n",
    "        return 'Low Risk'\n",
    "    elif prob < 0.50:\n",
    "        return 'Moderate Risk'\n",
    "    elif prob < 0.75:\n",
    "        return 'High Risk'\n",
    "    else:\n",
    "        return 'Very High Risk'\n",
    "\n",
    "results_df['Risk_Level'] = results_df['Deterioration_Probability'].apply(categorize_risk)\n",
    "results_df['Prediction_Text'] = results_df['Deterioration_Predicted'].map({0: 'No', 1: 'Yes'})\n",
    "\n",
    "print('Sample of Patient Predictions:')\n",
    "display(results_df[['True_Label', 'Deterioration_Probability', 'Risk_Level', 'Prediction_Text']].head(20))\n",
    "\n",
    "print('\\nRisk Level Distribution in Validation Set:')\n",
    "print(results_df['Risk_Level'].value_counts())\n"
]

new_cell = {
    "cell_type": "code",
    "metadata": {},
    "source": new_cell_source,
    "execution_count": None,
    "outputs": []
}

# Only append if we haven't already
already_added = False
for cell in nb["cells"]:
    if "Risk_Level" in "".join(cell.get("source", [])):
        already_added = True
        break

if not already_added:
    nb["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["### Patient Risk Stratification\n", "Categorizing the probabilities into low, moderate, high, and very high risk levels."]
    })
    nb["cells"].append(new_cell)
    
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print("Notebook updated.")
else:
    print("Notebook already contains Risk Level code.")
