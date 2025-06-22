
# Tesla Stock Price Forecasting (ML Pipeline)


A machine learning pipeline for forecasting Tesla's closing stock prices using:
- Linear Regression
- Support Vector Machine (SVM)
- LSTM (optional extension)

This project includes:
✅ Data preprocessing  
✅ Sliding window supervised dataset creation  
✅ Model training & evaluation  
✅ Recursive future prediction  
✅ Pytest unit tests  
✅ GitHub Actions continuous integration  

---

## Evaluation Metrics:
```
Mean Absolute Error (MAE)
Root Mean Squared Error (RMSE)
Visual Actual vs Predicted plots
Model comparison bar charts
Hyperparameter tuning with GridSearchCV and KerasTuner

## 📂 Project structure
```
my-tesla-ml-project/
├── .github/workflows/python-tests.yml     # GitHub Actions CI workflow
├── tests/test_ml_pipeline.py              # Pytest unit tests
├── requirements.txt                       # Project dependencies
├── src/project_tesla.py                   # code 
└── README.md                              # Project documentation
```

---

## ⚙️ Setup

### Clone the repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Run the pipeline
You can run your ML code from `src/` (or main notebook/script).  
Example:
```python
python src/your_ml_code.py
```

---

## 🧪 Run tests
To run tests locally:
```bash
pytest -v tests/
```

GitHub Actions will run tests automatically on push / PR.

---

## 📈 Example output
- **Plots**: Actual vs predicted prices  
- **Metrics**: MAE, RMSE reported for test data  
- **Future forecasts**: Recursive predictions for future closing prices  

---

## 🔑 Notes
- Ensure you have access to the Tesla stock CSV dataset (e.g. `TSLA.csv`).  
- The pipeline assumes time series order (no shuffle in train/test split).

---

## 🛠 TODO
- Add more features (Open, High, Low, Volume)
- Add stacked models / ensembles
- Add hyperparameter tuning for LSTM


---

