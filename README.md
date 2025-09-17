# Stock_Price_Forecasting
# Stock Price Prediction using Machine Learning and Deep Learning

## Project Overview
This project focuses on predicting stock prices of **16 Indian banking stocks** fetched using **yfinance**.  
The dataset was preprocessed with **Min-Max Scaling (per ticker)** and **log transformation on volume** to handle scale differences.  
Multiple **Machine Learning** and **Deep Learning** models were trained, with hyperparameter tuning performed using **Optuna** to achieve the best results.

## Models Used
- **Machine Learning Models**
  - Linear Regression (LR)
  - Random Forest Regressor (RFR)
  - Gradient Boosting Regressor (GBR)
  - XGBoost Regressor (XGBR)
  - LightGBM Regressor (LGBMR)
  - CatBoost Regressor (CATR)

- **Deep Learning Models**
  - Long Short-Term Memory (LSTM)
  - Gated Recurrent Unit (GRU)

## Key Features
- Real-world stock data fetched using **yfinance**.
- Feature preprocessing: **Min-Max scaling (per ticker)** and **log transformation on volume**.
- Hyperparameter optimization with **Optuna** for each model.
- Training and evaluation of models on optimized parameters.
- Comparison of ML vs. DL model performance.

## Evaluation Metrics
- **R² Score**
- **Adjusted R² Score**
- **Mean Absolute Error (MAE)**
- **Mean Absolute Percentage Error (MAPE)**
- **Root Mean Squared Error (RMSE)**

## Deliverables
- Best hyperparameters for each model (found using Optuna).
- Trained models on optimized parameters.
- Performance comparison of ML vs. DL approaches.
- Achieved highest **R² score of 0.93 with LightGBM**.

## Tech Stack
- **Python**
- **yfinance** (for data collection)
- **Pandas, NumPy, Scikit-learn**
- **XGBoost, LightGBM, CatBoost**
- **TensorFlow / Keras**
- **Optuna** (for hyperparameter tuning)
- **Matplotlib, Seaborn** (for visualization)

---
*This project demonstrates the effectiveness of feature engineering and hyperparameter tuning in improving stock price prediction using real-world data.*
