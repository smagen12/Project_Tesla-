
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

df = pd.read_csv('/content/drive/My Drive/TSLA.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
scaler = MinMaxScaler()
df['Close_Scaled'] = scaler.fit_transform(df[['Close']])
window_size = 3

def create_supervised_data(data, window=3):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i-window:i])
        y.append(data[i])
    return np.array(X), np.array(y)

X, y = create_supervised_data(df['Close_Scaled'].values, window=window_size)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
pred_scaled = model_lr.predict(X_test)
pred_unscaled = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

def test_close_scaled_range():
    assert df['Close_Scaled'].min() >= 0
    assert df['Close_Scaled'].max() <= 1

def test_supervised_data_shapes():
    assert X.shape[0] == len(df) - window_size
    assert X.shape[1] == window_size
    assert len(X) == len(y)

def test_prediction_length():
    assert len(pred_unscaled) == len(y_test_unscaled)

def test_future_prediction_length():
    num_future_predictions = 10
    last_window_scaled = df['Close_Scaled'].values[-window_size:]
    future_preds_scaled = []
    current_window = last_window_scaled.copy()
    for _ in range(num_future_predictions):
        input_data = current_window.reshape(1, -1)
        pred = model_lr.predict(input_data)[0]
        future_preds_scaled.append(pred)
        current_window = np.append(current_window[1:], pred)
    future_unscaled = scaler.inverse_transform(np.array(future_preds_scaled).reshape(-1, 1)).flatten()
    assert len(future_unscaled) == num_future_predictions
    assert (future_unscaled > 0).all()

def test_inverse_scaler_restores_close():
    first_close = df['Close'].iloc[0]
    scaled = scaler.transform([[first_close]])
    restored = scaler.inverse_transform(scaled)[0,0]
    assert np.isclose(restored, first_close, atol=1e-3)
