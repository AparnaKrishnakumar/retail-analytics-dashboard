"""
ML Model Training - Sales Forecasting
Using Prophet for time series prediction
"""

import pandas as pd
import numpy as np
from prophet import Prophet
import pickle
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

def load_processed_data():
    """Load cleaned data from pipeline"""
    df = pd.read_csv('data/processed/clean_data.csv')
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    print(f" Loaded {len(df):,} records")
    return df

def prepare_forecast_data(df):
    """
    Prepare data for Prophet model
    Prophet requires columns named 'ds' (date) and 'y' (value)
    """
    # Aggregate daily sales
    daily_sales = df.groupby(df['InvoiceDate'].dt.date)['TotalSales'].sum().reset_index()
    daily_sales.columns = ['ds', 'y']
    
    # Convert to datetime
    daily_sales['ds'] = pd.to_datetime(daily_sales['ds'])
    
    print(f" Prepared {len(daily_sales)} days of data")
    print(f"  Date range: {daily_sales['ds'].min()} to {daily_sales['ds'].max()}")
    print(f"  Avg daily sales: ${daily_sales['y'].mean():,.2f}")
    
    return daily_sales

def split_train_test(df, test_days=30):
    """Split data into train/test sets"""
    split_date = df['ds'].max() - timedelta(days=test_days)
    
    train = df[df['ds'] <= split_date]
    test = df[df['ds'] > split_date]
    
    print(f"\n Data split:")
    print(f"  Training: {len(train)} days")
    print(f"  Testing: {len(test)} days")
    
    return train, test

def train_prophet_model(train_data):
    """
    Train Prophet model with custom parameters
    """
    print("\n Training Prophet model...")
    
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='multiplicative',  # Better for retail data
        changepoint_prior_scale=0.05  # Controls flexibility
    )
    
    model.fit(train_data)
    
    print(" Model training complete")
    return model

def evaluate_model(model, test_data):
    """Calculate prediction accuracy"""
    # Make predictions for test period
    forecast = model.predict(test_data[['ds']])
    
    # Merge with actual values
    comparison = test_data.merge(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds')
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    comparison['ape'] = abs(comparison['y'] - comparison['yhat']) / comparison['y']
    mape = comparison['ape'].mean() * 100
    
    # Calculate RMSE
    rmse = np.sqrt(((comparison['y'] - comparison['yhat']) ** 2).mean())
    
    print(f"\n Model Performance:")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  RMSE: ${rmse:,.2f}")
    
    return comparison, mape, rmse

def save_model(model, filepath='models/sales_forecast.pkl'):
    """Save trained model"""
    import os
    os.makedirs('models', exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"\n✓ Model saved to {filepath}")

def generate_future_forecast(model, periods=30):
    """Generate forecast for next N days"""
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    
    # Get just the future predictions
    future_forecast = forecast.tail(periods)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    
    print(f"\n Future Forecast ({periods} days):")
    print(f"  Predicted total: ${future_forecast['yhat'].sum():,.2f}")
    print(f"  Daily average: ${future_forecast['yhat'].mean():,.2f}")
    
    return forecast

def main():
    """Run complete training pipeline"""
    print("=" * 50)
    print("SALES FORECASTING MODEL TRAINING")
    print("=" * 50)
    
    # Load data
    df = load_processed_data()
    
    # Add TotalSales if not present
    if 'TotalSales' not in df.columns:
        df['TotalSales'] = df['Quantity'] * df['UnitPrice']
    
    # Prepare for Prophet
    daily_sales = prepare_forecast_data(df)
    
    # Split data
    train, test = split_train_test(daily_sales, test_days=30)
    
    # Train model
    model = train_prophet_model(train)
    
    # Evaluate
    comparison, mape, rmse = evaluate_model(model, test)
    
    # Save results
    comparison.to_csv('data/output/model_evaluation.csv', index=False)
    print(" Evaluation results saved to data/output/model_evaluation.csv")
    
    # Generate future forecast
    forecast = generate_future_forecast(model, periods=30)
    forecast.to_csv('data/output/forecast_30days.csv', index=False)
    print(" Forecast saved to data/output/forecast_30days.csv")
    
    # Save model
    save_model(model)
    
    print("\n" + "=" * 50)
    print(" TRAINING COMPLETE")
    print(f" Model MAPE: {mape:.2f}%")
    print("=" * 50)
    
    return model, forecast, mape

if __name__ == "__main__":
    model, forecast, mape = main()