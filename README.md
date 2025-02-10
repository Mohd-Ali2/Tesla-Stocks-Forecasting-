# TSLA Stock Price Prediction with LSTM

## Overview
This project focuses on analyzing and predicting Tesla (TSLA) stock prices using historical data and a Long Short-Term Memory (LSTM) model. The dataset used contains historical stock prices, and the approach involves data preprocessing, feature engineering, and deep learning with PyTorch.

## Dataset
The dataset used in this project is `TSLA.csv`, which includes the following columns:
- `Date`: The trading date.
- `Open`: The opening price.
- `High`: The highest price of the day.
- `Low`: The lowest price of the day.
- `Close`: The closing price.
- `Adj Close`: The adjusted closing price.
- `Volume`: The number of shares traded.

## Dependencies
To run this project, install the required Python libraries:
```bash
pip install numpy pandas matplotlib torch scikit-learn
```

## Steps Involved

### 1. Load and Preprocess Data
- Read the dataset using Pandas.
- Select relevant columns (`Date` and `Close` price).
- Convert `Date` column to datetime format.
- Plot the closing price trend.

### 2. Feature Engineering
- Create lag features to capture past closing prices as inputs for LSTM.
- Normalize the dataset using MinMaxScaler.
- Prepare training data with a specified lookback period.

### 3. Convert Data for Model Input
- Convert data into NumPy arrays.
- Reshape data into a format suitable for PyTorch.

### 4. Model Training (Upcoming Implementation)
- Define the LSTM model using `torch.nn`.
- Train the model on historical stock prices.
- Evaluate the model's performance and make predictions.

## Usage
Run the script to preprocess and visualize the stock price data:
```bash
python main.py
```
(Upcoming) Train the LSTM model and make predictions.

## Future Improvements
- Implement and train an LSTM model for stock price forecasting.
- Fine-tune hyperparameters to improve accuracy.
- Develop a Streamlit or Flask web app for real-time predictions.

## Author
Mohammad Ali

## License
This project is open-source and available under the MIT License.

