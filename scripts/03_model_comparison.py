"""
Model Comparison Framework
Train and compare multiple forecasting approaches
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

# ==================== MODEL 1: BASELINE (MOVING AVERAGE) ====================

def baseline_moving_average(train, test, window=7):
    """Simple moving average baseline"""
    ma = train['sales'].rolling(window=window).mean().iloc[-1]
    predictions = np.full(len(test), ma)
    
    mape = np.mean(np.abs((test['sales'] - predictions) / test['sales'])) * 100
    rmse = np.sqrt(np.mean((test['sales'] - predictions) ** 2))
    
    return {
        'name': f'Moving Average ({window}d)',
        'predictions': predictions,
        'mape': mape,
        'rmse': rmse
    }

# ==================== MODEL 2: EXPONENTIAL SMOOTHING ====================

def exponential_smoothing(train, test):
    """Exponential Smoothing (Holt-Winters)"""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    
    try:
        model = ExponentialSmoothing(
            train['sales'],
            seasonal_periods=7,
            trend='add',
            seasonal='add'
        )
        fitted = model.fit()
        predictions = fitted.forecast(steps=len(test))
        
        mape = np.mean(np.abs((test['sales'] - predictions) / test['sales'])) * 100
        rmse = np.sqrt(np.mean((test['sales'] - predictions) ** 2))
        
        return {
            'name': 'Exponential Smoothing',
            'predictions': predictions.values,
            'mape': mape,
            'rmse': rmse,
            'model': fitted
        }
    except Exception as e:
        print(f"   Exponential Smoothing failed: {e}")
        return None

# ==================== MODEL 3: ARIMA ====================

def arima_model(train, test):
    """ARIMA model"""
    from statsmodels.tsa.arima.model import ARIMA
    
    try:
        # Auto-find best parameters (simple version)
        model = ARIMA(train['sales'], order=(1, 1, 1))
        fitted = model.fit()
        predictions = fitted.forecast(steps=len(test))
        
        mape = np.mean(np.abs((test['sales'] - predictions) / test['sales'])) * 100
        rmse = np.sqrt(np.mean((test['sales'] - predictions) ** 2))
        
        return {
            'name': 'ARIMA(1,1,1)',
            'predictions': predictions.values,
            'mape': mape,
            'rmse': rmse,
            'model': fitted
        }
    except Exception as e:
        print(f"   ARIMA failed: {e}")
        return None

# ==================== MODEL 4: PROPHET ====================

def prophet_model(train, test):
    """Facebook Prophet"""
    from prophet import Prophet
    
    try:
        # Prepare data
        train_prophet = pd.DataFrame({
            'ds': train.index,
            'y': train['sales'].values
        })
        
        # Train
        model = Prophet(
            yearly_seasonality=False,  # Not enough data
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode='additive'
        )
        model.fit(train_prophet)
        
        # Predict
        future = pd.DataFrame({'ds': test.index})
        forecast = model.predict(future)
        predictions = forecast['yhat'].values
        
        mape = np.mean(np.abs((test['sales'] - predictions) / test['sales'])) * 100
        rmse = np.sqrt(np.mean((test['sales'] - predictions) ** 2))
        
        return {
            'name': 'Prophet',
            'predictions': predictions,
            'mape': mape,
            'rmse': rmse,
            'model': model
        }
    except Exception as e:
        print(f"   Prophet failed: {e}")
        return None

# ==================== MODEL 5: LINEAR REGRESSION ====================

def linear_regression_model(train, test):
    """Linear Regression with time features"""
    from sklearn.linear_model import LinearRegression
    
    # Prepare features
    def create_features(df):
        df = df.copy()
        df['day_num'] = np.arange(len(df))
        df['dayofweek'] = df.index.dayofweek
        df['month'] = df.index.month
        return df[['day_num', 'dayofweek', 'month']]
    
    X_train = create_features(train)
    y_train = train['sales']
    
    X_test = create_features(test)
    X_test['day_num'] += len(train)  # Continue sequence
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    mape = np.mean(np.abs((test['sales'] - predictions) / test['sales'])) * 100
    rmse = np.sqrt(np.mean((test['sales'] - predictions) ** 2))
    
    return {
        'name': 'Linear Regression',
        'predictions': predictions,
        'mape': mape,
        'rmse': rmse,
        'model': model
    }

# ==================== MAIN COMPARISON ====================

def main():
    print("=" * 60)
    print("MODEL COMPARISON FRAMEWORK")
    print("=" * 60)
    
    # Load cleaned data
    print("\n1. Loading data...")
    df = pd.read_csv('data/processed/daily_sales_clean.csv', index_col=0, parse_dates=True)
    print(f"   {len(df)} days loaded")
    
    # Train/test split
    print("\n2. Splitting data...")
    test_size = 30
    train = df.iloc[:-test_size]
    test = df.iloc[-test_size:]
    print(f"   Train: {len(train)} days")
    print(f"   Test: {len(test)} days")
    
    # Train all models
    print("\n3. Training models...")
    results = []
    
    print("\n   → Baseline Moving Average...")
    results.append(baseline_moving_average(train, test, window=7))
    
    print("   → Exponential Smoothing...")
    es_result = exponential_smoothing(train, test)
    if es_result:
        results.append(es_result)
    
    print("   → ARIMA...")
    arima_result = arima_model(train, test)
    if arima_result:
        results.append(arima_result)
    
    print("   → Prophet...")
    prophet_result = prophet_model(train, test)
    if prophet_result:
        results.append(prophet_result)
    
    print("   → Linear Regression...")
    results.append(linear_regression_model(train, test))
    
    # Compare results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    comparison_df = pd.DataFrame([
        {'Model': r['name'], 'MAPE (%)': r['mape'], 'RMSE ($)': r['rmse']}
        for r in results
    ])
    comparison_df = comparison_df.sort_values('MAPE (%)')
    
    print(comparison_df.to_string(index=False))
    
    # Winner
    best_model = comparison_df.iloc[0]
    print(f"\n🏆 BEST MODEL: {best_model['Model']}")
    print(f"   MAPE: {best_model['MAPE (%)']:.2f}%")
    print(f"   RMSE: ${best_model['RMSE ($)']:,.2f}")
    
    # Save results
    comparison_df.to_csv('data/output/model_comparison.csv', index=False)
    print("\n✓ Results saved to data/output/model_comparison.csv")
    
    # Save best model predictions
    best_idx = comparison_df.index[0]
    best_predictions = results[best_idx]['predictions']
    
    forecast_df = pd.DataFrame({
        'date': test.index,
        'actual': test['sales'].values,
        'predicted': best_predictions
    })
    forecast_df.to_csv('data/output/best_model_forecast.csv', index=False)
    
    return results, comparison_df

if __name__ == "__main__":
    results, comparison = main()