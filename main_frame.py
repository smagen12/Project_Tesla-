import pandas as pd 
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import keras_tuner as kt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional
from statsmodels.tsa.arima.model import ARIMA


# Load the stock data
file_path = r'C:\Users\maayi\Downloads\TSLA.csv'
data = pd.read_csv(file_path)
# print(data.head())
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date')

#resample data 
monthly_data = data.resample('ME', on='Date').agg({
    'High': 'max',
    'Low': 'min',
    'Open': 'mean',
    'Close': 'last',
    'Volume': 'sum'
}).reset_index()

# drop values with nan values 
monthly_data .dropna(inplace=True)

# monthly return 
monthly_data['Monthly_Return'] = monthly_data['Close'].pct_change()

#Moving Averages (5, 10, 20 months)
monthly_data['MA5'] = monthly_data['Close'].rolling(window=5).mean()
monthly_data['MA10'] = monthly_data['Close'].rolling(window=10).mean()
monthly_data['MA20'] = monthly_data['Close'].rolling(window=20).mean()

# volatility 
monthly_data['Volatility_3'] = monthly_data['Close'].rolling(window=3).std()

# normalizing and standarize data 
features_to_scale = ['Close', 'Volume', 'Monthly_Return', 'MA5', 'MA10', 'MA20', 'Volatility_3']
scaler = StandardScaler()
monthly_data[features_to_scale] = scaler.fit_transform(monthly_data[features_to_scale])
 
#save the file 
output_path = r'S:\Internship\project 1\tesla_monthly_clean.csv'
monthly_data.to_csv(output_path, index=False)
print("data ready")

# Load your feature-enhanced, cleaned dataset
file_path = r'S:\Internship\project 1\tesla_monthly_clean.csv'
df = pd.read_csv(file_path)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# Closing Price Over Time
plt.figure(figsize=(12, 5))
plt.plot(df['Date'], df['Close'], label='Closing Price')
plt.title('Tesla Monthly Closing Price Trend')
plt.xlabel('Date')
plt.ylabel('Scaled Closing Price')
plt.grid(True)
plt.legend()
plt.show()

# Volatility Plot
plt.figure(figsize=(12, 5))
plt.plot(df['Date'], df['Volatility_3'], label='Volatility (3-month rolling std)')
plt.title('Volatility Over Time')
plt.xlabel('Date')
plt.ylabel('Volatility (Standard Deviation)')
plt.grid(True)
plt.legend()
plt.show()

# Volume vs Price Movement
plt.figure(figsize=(12, 5))
sns.scatterplot(x='Volume', y='Monthly_Return', data=df)
plt.title('Monthly Return vs Volume')
plt.xlabel('Volume')
plt.ylabel('Monthly Return')
plt.grid(True)
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title('Feature Correlation Heatmap')
plt.show()

# spike/crash detection
plt.figure(figsize=(12, 5))
sns.boxplot(x=df['Monthly_Return'])
plt.title('Boxplot of Monthly Returns (crash Detection)')
plt.grid(True)
plt.show()

# Large spikes detection 
threshold = 2.5  # Customize threshold
outliers = df[abs(df['Monthly_Return']) > threshold]
print(" Potential spike:\n", outliers[['Date', 'Monthly_Return']])

# converting to supervised format 
def create_supervised_data(df, target_col='Close', window=3):
    X, y = [], []
    for i in range(window, len(df)):
        X.append(df[target_col].values[i-window:i])
        y.append(df[target_col].values[i])
    return pd.DataFrame(X), pd.Series(y)

X, y = create_supervised_data(df, window=3)

# Linear regression 
X, y = create_supervised_data(df, 'Close', window=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
pred_lr = model_lr.predict(X_test)

# ARIMA model 
# Load your cleaned data
df = pd.read_csv(r'S:\Internship\project 1\tesla_monthly_clean.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# Use unscaled closing price (you must use original scale for ARIMA)
close_data = df['Close']

# Train ARIMA model (order can be tuned)
model = ARIMA(close_data, order=(5, 1, 0))  # (p=5, d=1, q=0) is a common starting point
model_fit = model.fit()

# Forecast next 12 months
forecast_steps = 12
forecast = model_fit.forecast(steps=forecast_steps)

# Create time index for future predictions
last_date = df['Date'].iloc[-1]
forecast_dates = pd.date_range(last_date, periods=forecast_steps + 1, freq='M')[1:]

# Plot actual vs forecast
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], close_data, label='Historical Closing Price')
plt.plot(forecast_dates, forecast, label='ARIMA Forecast', marker='o', linestyle='--')
plt.title('ARIMA Forecast for Tesla Closing Price')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



# Random Forest Regressor
model_rf = RandomForestRegressor(n_estimators=100)
model_rf.fit(X_train, y_train)
pred_rf = model_rf.predict(X_test)

# SVM model 
# Train-test split
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

svm_model = SVR()
svm_params = {'C': [0.1, 1, 10],'gamma': [0.01, 0.1, 1],'kernel': ['rbf'] }

grid_search = GridSearchCV(svm_model, svm_params, scoring='neg_mean_squared_error', cv=3)
grid_search.fit(X_train, y_train)

best_svm = grid_search.best_estimator_
svm_predictions = best_svm.predict(X_test)

#evaluation 
mae = mean_absolute_error(y_test, svm_predictions)
rmse = np.sqrt(mean_squared_error(y_test, svm_predictions))
print("SVM with GridSearch Results:")
print(f"Best Params: {grid_search.best_params_}")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")

# LSTM model 
#Load and Prepare Dataset
df = pd.read_csv(r'S:\Internship\project 1\tesla_monthly_clean.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

#Scale Only the 'Close' Column
close_scaler = MinMaxScaler()
df['Close_Scaled'] = close_scaler.fit_transform(df[['Close']])

#Create Sliding Window Sequences
def create_sequence(data, window):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i-window:i])
        y.append(data[i])
    return np.array(X), np.array(y)

window_size = 3
X, y = create_sequence(df['Close_Scaled'].values, window_size)

#Train-Test Split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

#Reshape Input for LSTM
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

#Build and Train LSTM Model
model_lstm = Sequential()
model_lstm.add(LSTM(64, activation='relu', input_shape=(window_size, 1)))
model_lstm.add(Dense(1))
model_lstm.compile(optimizer='adam', loss='mse')

print("Training LSTM...")
model_lstm.fit(X_train, y_train, epochs=100, batch_size=8, verbose=0)

# Make Predictions and Inverse Transform 
pred_lstm = model_lstm.predict(X_test)
pred_lstm_unscaled = close_scaler.inverse_transform(pred_lstm)
y_test_unscaled = close_scaler.inverse_transform(y_test.reshape(-1, 1))

# Evaluation
mae = mean_absolute_error(y_test_unscaled, pred_lstm_unscaled)
rmse = np.sqrt(mean_squared_error(y_test_unscaled, pred_lstm_unscaled))
print(" LSTM Forecast Results")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")

# Plot Actual vs Predicted 
plt.figure(figsize=(12, 5))
plt.plot(y_test_unscaled, label='Actual', marker='o')
plt.plot(pred_lstm_unscaled, label='LSTM Predicted', marker='x')
plt.title('LSTM - Actual vs Predicted Closing Prices')
plt.xlabel('Time Index')
plt.ylabel('Tesla Close Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Bidirectional LSTM
# Load and Prepare Dataset 
df = pd.read_csv(r'S:\Internship\project 1\tesla_monthly_clean.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# Scale Only the 'Close' Column 
close_scaler = MinMaxScaler()
df['Close_Scaled'] = close_scaler.fit_transform(df[['Close']])

#Create Sliding Window Sequences
def create_sequence(data, window):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i-window:i])
        y.append(data[i])
    return np.array(X), np.array(y)

window_size = 3
X, y = create_sequence(df['Close_Scaled'].values, window=window_size)

# Train-Test Split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Reshape Input for LSTM 
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Build and Train model
model_bilstm = Sequential()
model_bilstm.add(Bidirectional(LSTM(64, activation='relu'), input_shape=(window_size, 1)))
model_bilstm.add(Dense(1))
model_bilstm.compile(optimizer='adam', loss='mse')

print("Training Bidirectional LSTM...")
model_bilstm.fit(X_train, y_train, epochs=100, batch_size=8, verbose=0)

# Make Predictions and Inverse Transform 
pred_bilstm = model_bilstm.predict(X_test)
pred_bilstm_unscaled = close_scaler.inverse_transform(pred_bilstm)
y_test_unscaled = close_scaler.inverse_transform(y_test.reshape(-1, 1))

# Evaluation 
mae_bilstm = mean_absolute_error(y_test_unscaled, pred_bilstm_unscaled)
rmse_bilstm = np.sqrt(mean_squared_error(y_test_unscaled, pred_bilstm_unscaled))

print(" Bidirectional LSTM Forecast Results")
print(f"MAE  : {mae_bilstm:.2f}")
print(f"RMSE : {rmse_bilstm:.2f}")

# Plot Actual vs Predicted 
plt.figure(figsize=(12, 5))
plt.plot(y_test_unscaled, label='Actual', marker='o')
plt.plot(pred_bilstm_unscaled, label='BiLSTM Predicted', marker='x')
plt.title('BiLSTM - Actual vs Predicted Tesla Closing Prices')
plt.xlabel('Time Index')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Hyperparameter Tuning for LSTM & BiLSTM
# Load and scale only 'Close' column
df = pd.read_csv(r'S:\Internship\project 1\tesla_monthly_clean.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

scaler = MinMaxScaler()
df['Close_Scaled'] = scaler.fit_transform(df[['Close']])

# Create sliding window sequence
def create_sequence(data, window):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i-window:i])
        y.append(data[i])
    return np.array(X), np.array(y)

window_size = 3
X, y = create_sequence(df['Close_Scaled'].values, window=window_size)

# Train-test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# for LSTM
def build_lstm_model(hp):
    model = Sequential()
    model.add(LSTM(
        units=hp.Choice('units', [32, 64, 128]),
        activation=hp.Choice('activation', ['relu', 'tanh']),
        input_shape=(X_train.shape[1], 1)
    ))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

tuner = kt.RandomSearch(
    build_lstm_model,
    objective='val_loss',
    max_trials=5,
    executions_per_trial=1,
    directory='tuning',
    project_name='lstm_tuning'
)

tuner.search(X_train, y_train, epochs=50, validation_split=0.2, verbose=1)


#  for BiLSTM
def build_bilstm_model(hp):
    model = Sequential()
    model.add(Bidirectional(LSTM(
        units=hp.Choice('units', [32, 64, 128]),
        activation=hp.Choice('activation', ['relu', 'tanh'])
    ), input_shape=(X_train.shape[1], 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

tuner = kt.RandomSearch(
    build_bilstm_model,
    objective='val_loss',
    max_trials=5,
    executions_per_trial=1,
    directory='tuning',
    project_name='bilstm_tuning'
)

tuner.search(X_train, y_train, epochs=50, validation_split=0.2, verbose=1)

# comparing both models 
best_model = tuner.get_best_models(num_models=1)[0]
pred = best_model.predict(X_test)
pred_unscaled = scaler.inverse_transform(pred)
y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))

mae = np.mean(np.abs(y_test_unscaled - pred_unscaled))
rmse = np.sqrt(mean_squared_error(y_test_unscaled, pred_unscaled))
print(f"Best Model MAE: {mae:.2f}, RMSE: {rmse:.2f}")  

# Retrieve Best Tuned Models (LSTM and BiLSTM models)
best_lstm_model = tuner.get_best_models(num_models=1)[0]
pred_lstm_tuned = best_lstm_model.predict(X_test)

# Rebuild and rerun for BiLSTM separately to avoid overwriting 'tuner'
tuner_bilstm = kt.RandomSearch(
    build_bilstm_model,
    objective='val_loss',
    max_trials=5,
    executions_per_trial=1,
    directory='tuning',
    project_name='bilstm_tuning'
)
tuner_bilstm.reload()
best_bilstm_model = tuner_bilstm.get_best_models(num_models=1)[0]
pred_bilstm_tuned = best_bilstm_model.predict(X_test)

# Inverse Transform 
pred_lstm_unscaled = scaler.inverse_transform(pred_lstm_tuned).flatten()
pred_bilstm_unscaled = scaler.inverse_transform(pred_bilstm_tuned).flatten()
y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

# Evaluate 
mae_lstm_tuned = mean_absolute_error(y_test_unscaled, pred_lstm_unscaled)
rmse_lstm_tuned = np.sqrt(mean_squared_error(y_test_unscaled, pred_lstm_unscaled))
mae_bilstm_tuned = mean_absolute_error(y_test_unscaled, pred_bilstm_unscaled)
rmse_bilstm_tuned = np.sqrt(mean_squared_error(y_test_unscaled, pred_bilstm_unscaled))

# Plot
plt.figure(figsize=(12, 6))
plt.plot(y_test_unscaled, label='Actual', marker='o')
plt.plot(pred_lstm_unscaled, label=f'LSTM Tuned (MAE: {mae_lstm_tuned:.2f}, RMSE: {rmse_lstm_tuned:.2f})', marker='x')
plt.plot(pred_bilstm_unscaled, label=f'BiLSTM Tuned (MAE: {mae_bilstm_tuned:.2f}, RMSE: {rmse_bilstm_tuned:.2f})', marker='^')
plt.title('Tuned LSTM vs BiLSTM - Actual vs Predicted')
plt.xlabel('Time Index')
plt.ylabel('Tesla Stock Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Custom RMSE function
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Truncate actual values to match predictions (LSTM predictions are usually shortest)
min_len = min(len(pred_lr), len(pred_rf), len(svm_predictions), len(pred_lstm_unscaled), len(pred_bilstm_unscaled))
y_actual = y_test_unscaled[-min_len:]  # unscaled actual values for LSTM & BiLSTM

# List of model names, predictions, and best models 
model_names = [
    'Linear Regression',
    'Random Forest',
    'Support Vector Machines (SVM)',
    'LSTM',
    'Bidirectional LSTM (BiLSTM)'
]

predictions = [
    pred_lr[-min_len:],          
    pred_rf[-min_len:],
    svm_predictions[-min_len:],
    pred_lstm_unscaled[-min_len:],
    pred_bilstm_unscaled[-min_len:]
]

best_models = [
    model_lr,
    model_rf,
    best_svm,
    model_lstm,
    model_bilstm
]

# Evaluate and display
for i, model_name in enumerate(model_names):
    y_pred = predictions[i]
    model = best_models[i]

    model_rmse = rmse(y_actual, y_pred)
    model_mae = mean_absolute_error(y_actual, y_pred)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(y_actual, label='Actual', marker='o')
    plt.plot(y_pred, label='Predicted', marker='x')
    plt.title(f"{model_name}\nMAE: {model_mae:.2f} | RMSE: {model_rmse:.2f}")
    plt.xlabel('Time Index')
    plt.ylabel('Tesla Stock Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Print model details
    print(f"Model: {model_name}")
    if hasattr(model, 'get_params'):
        print("Best Parameters:")
        print(model.get_params())
    else:
        print("Model summary:")
        model.summary()
    print("-" * 40)
