# Sales Forecasting Model Documentation

## Overview
Time series forecasting model for predicting retail sales using Facebook Prophet.

## Model Details

### Algorithm: Prophet
- **Type**: Additive regression model
- **Components**: Trend + Seasonality + Holidays
- **Why Prophet**: Handles missing data, robust to outliers, captures multiple seasonality patterns

### Training Data
- **Period**: [Your date range]
- **Records**: [Number] daily observations
- **Features**: Date, Daily sales total

### Model Parameters
```python
Prophet(
    yearly_seasonality=True,   # Captures annual patterns
    weekly_seasonality=True,    # Captures day-of-week effects
    daily_seasonality=False,    # Not needed for daily aggregates
    seasonality_mode='multiplicative',  # Better for retail (% changes)
    changepoint_prior_scale=0.05  # Flexibility in trend changes
)
```

## Performance Metrics

### Test Set Performance
- **MAPE**: [Your value]%
- **RMSE**: $[Your value]
- **MAE**: $[Your value]

### Interpretation
- MAPE < 10%: Excellent
- MAPE 10-20%: Good
- MAPE 20-30%: Acceptable
- MAPE > 30%: Needs improvement

**Our model**: [Quality rating]

## Use Cases

### 1. Revenue Forecasting
Predict next 30 days of sales for financial planning

### 2. Inventory Management
Anticipate demand spikes to optimize stock levels

### 3. Marketing Campaign Planning
Identify low-sales periods for promotional activities

### 4. Anomaly Detection
Flag days with unusual sales patterns

## Limitations

1. **Historical dependency**: Requires 6+ months of data for best results
2. **Exogenous factors**: Doesn't account for marketing campaigns, competitor actions
3. **Black swan events**: Cannot predict unprecedented disruptions
4. **Granularity**: Daily forecasts; intraday patterns not captured

## Deployment

### Current Status
- ✅ Model trained and saved (`models/sales_forecast.pkl`)
- ✅ Validation complete
- ⏳ Dashboard integration (planned)
- ⏳ Automated retraining (future)

### How to Use

**Generate new forecast:**
```python
import pickle
with open('models/sales_forecast.pkl', 'rb') as f:
    model = pickle.load(f)

future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)
```

**Retrain with new data:**
```bash
python scripts/02_train_model.py
```

## Future Enhancements

- [ ] Add external regressors (holidays, promotions)
- [ ] Ensemble with XGBoost for improved accuracy
- [ ] Category-level forecasts (not just total sales)
- [ ] Real-time model updating
- [ ] Confidence interval calibration
- [ ] Automated hyperparameter tuning

## Model Monitoring

**Recommended checks:**
- Weekly: Compare predictions vs actuals
- Monthly: Retrain with latest data
- Quarterly: Full model evaluation and tuning

## References
- [Prophet Documentation](https://facebook.github.io/prophet/)
- [Time Series Forecasting Best Practices](https://www.microsoft.com/en-us/research/project/forecasting/)

---

*Last updated: November 24, 2025*
*Model version: 1.0*