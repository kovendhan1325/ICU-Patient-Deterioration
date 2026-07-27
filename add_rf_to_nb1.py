import json
import os

notebook_path = "Notebook_1_Load_Data.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

new_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Train Random Forest and Predict Risk Levels\n",
            "We will flatten the 3D data, train a model, and categorize patient risk into Low, Moderate, High, and Very High."
        ]
    },
    {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "from sklearn.ensemble import RandomForestClassifier\n",
            "from sklearn.metrics import classification_report, roc_auc_score\n",
            "import pandas as pd\n",
            "\n",
            "# Load validation data\n",
            "X_val = np.load(os.path.join(data_dir, 'X_val.npy'))\n",
            "y_val = np.load(os.path.join(data_dir, 'y_val.npy'))\n",
            "\n",
            "# Flatten data for Random Forest\n",
            "N_train, seq_len, n_features = X_train.shape\n",
            "N_val = X_val.shape[0]\n",
            "X_train_flat = X_train.reshape(N_train, seq_len * n_features)\n",
            "X_val_flat = X_val.reshape(N_val, seq_len * n_features)\n",
            "\n",
            "# Train the model\n",
            "print('Training Random Forest...')\n",
            "rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1)\n",
            "rf_model.fit(X_train_flat, y_train)\n",
            "\n",
            "# Predictions and Probabilities\n",
            "y_val_pred = rf_model.predict(X_val_flat)\n",
            "y_val_prob = rf_model.predict_proba(X_val_flat)[:, 1]\n",
            "\n",
            "print('ROC-AUC Score:', roc_auc_score(y_val, y_val_prob))\n"
        ],
        "execution_count": None,
        "outputs": []
    },
    {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Create DataFrame with Risk Levels\n",
            "results_df = pd.DataFrame({\n",
            "    'True_Label': y_val,\n",
            "    'Deterioration_Predicted': y_val_pred,\n",
            "    'Deterioration_Probability': y_val_prob\n",
            "})\n",
            "\n",
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
            "print('Sample of Patient Risk Predictions:')\n",
            "display(results_df[['True_Label', 'Deterioration_Probability', 'Risk_Level', 'Prediction_Text']].head(20))\n",
            "\n",
            "print('\\nRisk Level Distribution in Validation Set:')\n",
            "print(results_df['Risk_Level'].value_counts())\n"
        ],
        "execution_count": None,
        "outputs": []
    }
]

# Avoid adding multiple times
already_added = False
for cell in nb.get("cells", []):
    if "### Train Random Forest and Predict Risk Levels" in "".join(cell.get("source", [])):
        already_added = True
        break

if not already_added:
    nb.setdefault("cells", []).extend(new_cells)
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print("Notebook 1 updated.")
else:
    print("Notebook 1 already has this code.")
