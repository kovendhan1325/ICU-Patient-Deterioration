Write-Host "Running Notebook 3"
python -m nbconvert --to notebook --execute --inplace Notebook_3_Train_LR_XGBoost.ipynb
Write-Host "Running Notebook 4"
python -m nbconvert --to notebook --execute --inplace Notebook_4_Train_LR_XGBoost_Tuned.ipynb
Write-Host "Running Notebook 5"
python -m nbconvert --to notebook --execute --inplace Notebook_5_Train_LSTM.ipynb
Write-Host "Done running all models"
