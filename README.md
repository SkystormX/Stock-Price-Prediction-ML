# Stock Price Prediction with Machine Learning

## Overview
This project applies machine learning techniques to predict stock prices using historical market data. Key highlights:
- **Feature Engineering**: Moving averages, volatility, and lagged prices.
- **Modeling**: Ridge Regression with polynomial features.
- **Performance Evaluation**: Metrics like MSE and MAE via K-Fold Cross-Validation.

This project showcases my skills in:
- Python programming
- Machine learning
- Financial data analysis
- Data visualization

## Key Features
- **Feature Engineering**: Includes moving averages, lagged price values, and volatility calculations.
- **Modeling**: Implements Ridge Regression with polynomial features for enhanced prediction capabilities.
- **Performance Evaluation**: Employs K-Fold Cross-Validation to assess metrics like Mean Squared Error (MSE) and Mean Absolute Error (MAE).
- **Visualizations**:
  - Actual vs Predicted Prices
  - Residual Plots to analyze errors
  - Distribution of errors and cumulative error progression
 
## Actual vs Predicted Prices
![Actual vs Predicted Prices](actual_vs_predicted.png)

## Residual Plot
![Residual Plot](residual_plot.png)

## Residual Distribution
![Residual Distribution](residual_distribution.png)

## Cumulative Error Plot
![Cumulative Error Plot](cumulative_error_plot.png)


## Limitations
- Predictions are based solely on historical prices; no external factors (e.g., news sentiment) are considered.
- The model may not generalize well to highly volatile or illiquid stocks.
- Ridge Regression with polynomial features might overfit to noise in small datasets.

## Future Improvements

While the project demonstrates a solid foundation in stock price prediction using historical data, several enhancements could further improve its accuracy and usability:

1. **Incorporate External Data Sources**  
   - Integrate additional data such as financial news sentiment, earnings reports, and macroeconomic indicators to provide a more comprehensive model.  
   - Example: Use APIs like [Alpha Vantage](https://www.alphavantage.co/) or [NewsAPI](https://newsapi.org/) to fetch real-time data.

2. **Enhance Feature Engineering**  
   - Explore advanced features such as technical indicators (e.g., RSI, Bollinger Bands) or sector-specific metrics.  
   - Use lagged features for longer time horizons to capture trends more effectively.

3. **Experiment with Advanced Machine Learning Models**  
   - Implement models like Random Forest, Gradient Boosting (e.g., XGBoost, LightGBM), or deep learning approaches (e.g., LSTM for time-series data).  
   - Compare these methods with the current Ridge Regression to determine the best-performing approach.

4. **Optimize Hyperparameters**  
   - Utilize techniques like Grid Search or Bayesian Optimization to fine-tune model parameters for better performance.

5. **Improve Cross-Validation Strategy**  
   - Use `TimeSeriesSplit` instead of `KFold` Cross-Validation to account for the sequential nature of time-series data.  
   - This would better reflect the model's performance on unseen future data.

By implementing these improvements, the project could evolve into a robust and versatile stock prediction tool, bridging the gap between theoretical models and real-world applications.



