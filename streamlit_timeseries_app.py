import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input
import matplotlib.pyplot as plt
import datetime as dt

# ============================
# 1. Title and Description
# ============================
st.title("Electricity Consumption/Pricing Forecasting")
st.markdown(
    """
    This application allows you to upload time series data and forecast electricity consumption using **SARIMA** and **LSTM** models.
    
    Please note:
    - The data must contain **two columns only**:
      - One column with **Date** (in any valid date format)
      - One column with **Price** or **Electricity Consumption (kWh)**.
    
    Once you upload your data, select parameters, and view the results interactively!
    """
)

# ============================
# 2. File Upload Section
# ============================
st.sidebar.header("Upload Your Time Series Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    # ============================
    # 3. Load Data and Show Initial Rows
    # ============================
    data = pd.read_csv(uploaded_file, parse_dates=[0], index_col=0)
    data.index.freq = "MS"  # Explicitly set frequency to Monthly Start
    st.write("### Uploaded Data")
    st.write(data.head())

    # ============================
    # 4. Log Transformation of Data
    # ============================
    ts_ln = np.log(data.iloc[:, 0])

    # ============================
    # 5. Input Future Date
    # ============================
    future_date = st.sidebar.date_input("Select Future Date", dt.date(2025, 1, 1))
    steps_to_forecast = (future_date.year - data.index[-1].year) * 12 + (future_date.month - data.index[-1].month)

    # ============================
    # 6. Select Forecasting Model
    # ============================
    model_option = st.sidebar.selectbox("Select Model", ["SARIMA", "LSTM"])

    if steps_to_forecast > 0:  # Ensure valid future steps
        # ============================
        # 7. SARIMA Model - Fit and Forecast
        # ============================
        if model_option == "SARIMA":
            st.write("### SARIMA Model")

            # Parameters Input for SARIMA
            p = st.sidebar.number_input("AR Order (p)", value=1, min_value=0)
            d = st.sidebar.number_input("Differencing Order (d)", value=1, min_value=0)
            q = st.sidebar.number_input("MA Order (q)", value=1, min_value=0)
            P = st.sidebar.number_input("Seasonal AR Order (P)", value=1, min_value=0)
            D = st.sidebar.number_input("Seasonal Differencing Order (D)", value=1, min_value=0)
            Q = st.sidebar.number_input("Seasonal MA Order (Q)", value=1, min_value=0)
            T = st.sidebar.number_input("Seasonal Period (T)", value=12, min_value=1)

            # Fit the SARIMA model
            model = SARIMAX(
                ts_ln,
                order=(p, d, q),
                seasonal_order=(P, D, Q, T),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            sarima_model = model.fit(disp=False)

            # Forecast future values
            forecast = np.exp(sarima_model.forecast(steps=steps_to_forecast))

            # Display Forecast Visualization with Differentiated Colors
            st.write("### SARIMA Forecast")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(data.index, np.exp(ts_ln), label="Historical Data", color='blue')
            forecast_index = pd.date_range(start=data.index[-1], periods=steps_to_forecast + 1, freq='MS')[1:]
            ax.plot(forecast_index, forecast, label="Forecast", color='red')  # Red for predicted values
            ax.legend()
            st.pyplot(fig)

            # Display forecast values for SARIMA
            forecast_index = pd.date_range(start=data.index[-1], periods=steps_to_forecast + 1, freq='MS')[1:]
            forecast_df = pd.DataFrame({'Date': forecast_index, 'Forecast': forecast})
            st.write("### SARIMA Forecast Values")
            st.dataframe(forecast_df)

        # ============================
        # 8. LSTM Model - Fit and Forecast
        # ============================
        elif model_option == "LSTM":
            st.write("### LSTM Model")

            # Scale the data
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(ts_ln.values.reshape(-1, 1))

            # Create dataset function
            def create_dataset(data, look_back=12):
                X, Y = [], []
                for i in range(len(data) - look_back):
                    X.append(data[i:(i + look_back), 0])
                    Y.append(data[i + look_back, 0])
                return np.array(X), np.array(Y)

            # Create dataset
            look_back = 12
            X, Y = create_dataset(scaled_data, look_back)

            # Reshape input to be [samples, time steps, features]
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))

            # Build the LSTM model
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(look_back, 1)),
                LSTM(50),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mean_squared_error')

            # Train the model
            model.fit(X, Y, epochs=100, batch_size=32, verbose=0)

            # Generate future predictions
            last_sequence = scaled_data[-look_back:]
            future_predictions = []

            for _ in range(steps_to_forecast):
                next_pred = model.predict(last_sequence.reshape(1, look_back, 1))
                future_predictions.append(next_pred[0, 0])
                last_sequence = np.append(last_sequence[1:], next_pred)

            # Inverse transform predictions
            future_predictions = np.array(future_predictions).reshape(-1, 1)
            future_predictions = scaler.inverse_transform(future_predictions)
            future_predictions = np.exp(future_predictions)  # Reverse log transformation

            # Create future dates for predictions
            last_date = data.index[-1]
            future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=steps_to_forecast, freq='MS')

            # Visualize results
            st.write("### LSTM Forecast")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(data.index, data.iloc[:, 0], label='Historical Data', color='blue')
            ax.plot(future_dates, future_predictions, label='LSTM Forecast', color='green')
            ax.set_title('LSTM Forecast')
            ax.set_xlabel('Date')
            ax.set_ylabel('Electricity Consumption')
            ax.legend()
            st.pyplot(fig)

            # Display forecast values
            # Display forecast values for LSTM
            forecast_df = pd.DataFrame({'Date': future_dates, 'Forecast': future_predictions.flatten()})
            st.write("### LSTM Forecast Values")
            st.dataframe(forecast_df)

else:
    st.write("Upload a time series CSV file to get started.")
