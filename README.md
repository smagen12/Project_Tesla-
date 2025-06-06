# Tesla Stock Price Forecasting

This project aims to forecast the **next month's closing price** of Tesla (TSLA) stock using a mix of **traditional machine learning** and **deep learning** models.

---

## 🚀 Project Overview

- **Dataset**: Monthly aggregated Tesla stock data (Open, High, Low, Close, Volume)
- **Goal**: Predict the next month's **closing price**
- **Approach**:
  - Data preprocessing and feature engineering
  - Exploratory Data Analysis (EDA)
  - Train and evaluate models
  - Hyperparameter tuning using `GridSearchCV` and `KerasTuner`
  - Visual comparison and performance analysis

---

## 📁 Project Structure

```
.
├── data/
│   └── tesla_monthly_clean.csv
├── notebooks/
│   └── EDA_Tesla_Price.ipynb
├── scripts/
│   └── main.py
├── models/
│   └── (optional: saved models)
├── tuning/
│   └── (auto-generated by KerasTuner)
├── README.md
├── requirements.txt
├── .gitignore
```

---

## 🧠 Features Used

- Monthly Return (`pct_change`)
- Moving Averages: MA5, MA10, MA20
- Volatility (3-month rolling standard deviation)
- Scaled features using `StandardScaler` and `MinMaxScaler`

---

##  Models Implemented

| Type         | Models                             |
|--------------|-------------------------------------|
| Traditional  | Linear Regression, Random Forest, SVM, ARIMA  |
| Deep Learning| LSTM, Bidirectional LSTM (BiLSTM)     |

---

##  Evaluation Metrics

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Visual Actual vs Predicted plots
- Model comparison bar charts
- Hyperparameter tuning with `GridSearchCV` and `KerasTuner`

---

##  Results Summary

All models were compared side-by-side for accuracy. LSTM/BiLSTM generally captured trends better than traditional models, but no model strongly outperformed a moving average baseline due to Tesla's high volatility and limited features.

---

##  How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run main script
python scripts/main.py
```

For interactive analysis, open the notebook:
```
notebooks/EDA_Tesla_Price.ipynb
```
---

## 📌 Notes

This project avoids overfitting by using validation splits and regularization where applicable. The code is modular and extensible for future experiments with multi-variate time series, exogenous features, or alternative forecasting targets (e.g., return classification).