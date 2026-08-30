import json, os

NB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Notebook_6_Improved_Models.ipynb')

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

shap_markdown = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Cell 12: Explainable AI (XAI) using SHAP\n",
        "Generate SHAP explanations for the best-performing tree model (LightGBM) to understand which features drive the predictions."
    ]
}

shap_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import shap\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "print('='*60)\n",
        "print('Explainable AI (XAI) - SHAP Analysis for LightGBM')\n",
        "print('='*60)\n",
        "\n",
        "try:\n",
        "    # Initialize JS visualization for notebook\n",
        "    shap.initjs()\n",
        "    \n",
        "    # Reconstruct feature names for flattened input\n",
        "    n_timesteps = SEQUENCE_LENGTH\n",
        "    flat_names = []\n",
        "    for t in range(n_timesteps):\n",
        "        for f in feature_cols:\n",
        "            flat_names.append(f'h{t}_{f}')\n",
        "            \n",
        "    # Using a sample of test data for SHAP to save time\n",
        "    Xf_test_sample = Xf_test[:500]\n",
        "    \n",
        "    # Explain the LightGBM model\n",
        "    print('Calculating SHAP values (this may take a minute)...')\n",
        "    explainer = shap.TreeExplainer(lgb_model)\n",
        "    shap_values = explainer.shap_values(Xf_test_sample)\n",
        "    \n",
        "    # For LightGBM binary classification, shap_values is a list of length 2 (for class 0 and class 1)\n",
        "    # We want the explanations for the positive class (class 1)\n",
        "    if isinstance(shap_values, list):\n",
        "        shap_values_pos = shap_values[1]\n",
        "    else:\n",
        "        shap_values_pos = shap_values\n",
        "        \n",
        "    # 1. Summary Plot (Beeswarm)\n",
        "    plt.figure(figsize=(10, 6))\n",
        "    # Temporarily switch to default style for SHAP as it doesn't play well with dark backgrounds\n",
        "    plt.style.use('default') \n",
        "    shap.summary_plot(shap_values_pos, Xf_test_sample, feature_names=flat_names, show=False, max_display=15)\n",
        "    plt.title('SHAP Summary Plot (Top 15 Features)', fontsize=14, fontweight='bold', pad=20)\n",
        "    plt.tight_layout()\n",
        "    shap_path = os.path.join(BASE_DIR, 'preprocessing_pipeline', 'reports', 'shap_summary.png')\n",
        "    plt.savefig(shap_path, dpi=150, bbox_inches='tight')\n",
        "    plt.show()\n",
        "    print(f'SHAP summary plot saved to {shap_path}')\n",
        "    \n",
        "    # Revert to dark background for future plots\n",
        "    plt.style.use('dark_background')\n",
        "    \n",
        "except Exception as e:\n",
        "    print(f'Error generating SHAP explanations: {e}')"
    ]
}

nb['cells'].extend([shap_markdown, shap_code])

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print('Added SHAP cell to Notebook 6.')
