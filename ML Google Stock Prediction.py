# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Load the stock price data
# Replace 'file_path' with the location of your stock data CSV file
file_path = r'B:\Arif\Pandas Projects\GOOGLE.csv'
data = pd.read_csv(file_path)

# Feature Engineering
# Create moving averages and lag features to capture trends and past behavior
data['5_day_MA'] = data['Close'].rolling(window=5).mean()  # 5-day Moving Average
data['Volatility'] = data['Close'].pct_change().fillna(0)  # Daily percentage change (proxy for volatility)
data['10_day_MA'] = data['Close'].rolling(window=10).mean()  # 10-day Moving Average
data['Close_t-1'] = data['Close'].shift(1)  # Close price lagged by 1 day
data['Close_t-2'] = data['Close'].shift(2)  # Close price lagged by 2 days
data = data.dropna()  # Drop rows with NaN values created by rolling and shifting

# Define features (X) and target (y)
# Features are engineered metrics and lagged prices; the target is the next day's closing price
X = data[['5_day_MA', 'Volatility', '10_day_MA', 'Close_t-1', 'Close_t-2']]
y = data['Close']

# K-Fold Cross-Validation with Ridge Regression
# Using 5-fold CV to evaluate model performance on multiple subsets of data
kf = KFold(n_splits=5, shuffle=True, random_state=42)  # Shuffle for better data distribution
fold = 1  # Track fold number
mse_scores = []  # List to store Mean Squared Error for each fold
mae_scores = []  # List to store Mean Absolute Error for each fold

# Loop through each fold
for train_index, test_index in kf.split(X):
    # Split data into training and testing sets for the current fold
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Ridge Regression with Polynomial Features (degree=2)
    # The pipeline applies polynomial transformations followed by Ridge regression
    pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),  # Add polynomial terms
        ('ridge', Ridge(alpha=1.0))  # Ridge regression to prevent overfitting
    ])
    pipeline.fit(X_train, y_train)  # Train the model on the current fold
    y_pred = pipeline.predict(X_test)  # Predict the target values

    # Calculate performance metrics for the current fold
    mse = mean_squared_error(y_test, y_pred)  # Mean Squared Error
    mae = mean_absolute_error(y_test, y_pred)  # Mean Absolute Error
    mse_scores.append(mse)
    mae_scores.append(mae)

    # Print fold-specific metrics
    print(f"Fold {fold}: MSE = {mse}, MAE = {mae}")
    fold += 1

# Calculate average metrics across all folds
average_mse = np.mean(mse_scores)
average_mae = np.mean(mae_scores)
print(f"\nAverage MSE: {average_mse}")
print(f"Average MAE: {average_mae}")

# Visualization of Actual vs Predicted Prices (Last Fold)
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual Prices', alpha=0.7)
plt.plot(y_pred, label='Predicted Prices', alpha=0.7)
plt.title('Actual vs Predicted Prices (Last Fold)')
plt.xlabel('Index')
plt.ylabel('Price')
plt.legend()
plt.show()

# Residual Plot for the Last Fold
# Residual = Actual - Predicted; this shows the errors for each prediction
residuals = y_test.values - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(range(len(residuals)), residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')  # Reference line at zero
plt.title('Residual Plot')
plt.xlabel('Index')
plt.ylabel('Residuals')
plt.show()

# Residual Distribution
# Histogram to visualize the distribution of residuals
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=50, alpha=0.7)
plt.axvline(x=0, color='r', linestyle='--')  # Reference line at zero
plt.title('Distribution of Residuals')
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.show()

# Cumulative Error Plot
# Cumulative sum of absolute residuals to observe error accumulation
cumulative_error = np.cumsum(np.abs(residuals))
plt.figure(figsize=(10, 6))
plt.plot(cumulative_error, label='Cumulative Error', color='purple')
plt.title('Cumulative Error Plot')
plt.xlabel('Index')
plt.ylabel('Cumulative Error')
plt.legend()
plt.show()
