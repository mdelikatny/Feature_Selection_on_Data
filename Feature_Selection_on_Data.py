import os 
import pandas as pd
import numpy as np
import ta
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score


def feature_selection(tick, stock_data, macro_data):
    # Create a DataFrame for stock data
    stock_df = pd.DataFrame()
    stock_df['Open'] = stock_data['Open'][tick].fillna(0)
    stock_df['High'] = stock_data['High'][tick].fillna(0)
    stock_df['Low'] = stock_data['Low'][tick].fillna(0)
    stock_df['Close'] = stock_data['Close'][tick].fillna(0)
    stock_df['Adj Close'] = stock_data['Adj Close'][tick].fillna(0)
    stock_df['Volume'] = stock_data['Volume'][tick].fillna(0)
    
    # Add technical indicators using the ta library
    tech_df = ta.add_all_ta_features(stock_df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
    tech_df.drop(['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'], axis=1, inplace=True)
    
    # Combine technical indicators with macroeconomic data
    all_data = pd.concat([tech_df, macro_data], axis=1)
    all_data.drop(all_data.tail(1).index, inplace=True)
    
    # Prepare target variable (1 if stock price increases, 0 otherwise)
    adj_close = stock_data['Adj Close'][tick].fillna(0)
    adj_close_pct_change = adj_close.pct_change().fillna(0)
    target = np.where(adj_close_pct_change > 0, 1, 0)
    target = target[1:]
    
    # Split data into training and validation sets
    split_index = 2000
    X_train = all_data.iloc[:split_index, :]
    y_train = target[:split_index]
    X_valid = all_data.iloc[split_index:, :]
    y_valid = target[split_index:]
    
    # Use Extra Trees Classifier for feature selection
    model = ExtraTreesClassifier()
    model.fit(X_train, y_train)
    feature_importances = model.feature_importances_
    
    return feature_importances
    

# Example usage:
tickers = ['AAPL', 'GOOG', 'MSFT']
data = pd.read_csv('stock_data.csv') # replace with actual data file
macro_data = pd.read_csv('macro_data.csv') # replace with actual macro data file
all_results = []
for tick in tickers:
    result = feature_selection(tick, data, macro_data)
    all_results.append(result)

# Save results to CSV file
results_df = pd.DataFrame(all_results, columns=all_data.columns)
results_df.to_csv('feature_importances.csv', index=False)
