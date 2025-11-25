"""
Advanced Data Cleaning
Fixes issues identified in diagnosis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def load_and_diagnose():
    """Load data and identify issues"""
    df = pd.read_csv('data/processed/clean_data.csv')
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    if 'TotalSales' not in df.columns:
        df['TotalSales'] = df['Quantity'] * df['UnitPrice']
    
    return df

def handle_outliers(daily_sales, method='winsorize'):
    """Handle extreme outliers"""
    if method == 'winsorize':
        # Cap at 1st and 99th percentile
        lower = daily_sales.quantile(0.01)
        upper = daily_sales.quantile(0.99)
        daily_sales_clean = daily_sales.clip(lower=lower, upper=upper)
        print(f"   Winsorized: capped at ${lower:,.0f} - ${upper:,.0f}")
    
    elif method == 'remove':
        # Remove outliers using IQR
        Q1 = daily_sales.quantile(0.25)
        Q3 = daily_sales.quantile(0.75)
        IQR = Q3 - Q1
        mask = (daily_sales >= Q1 - 3*IQR) & (daily_sales <= Q3 + 3*IQR)
        daily_sales_clean = daily_sales[mask]
        print(f"   Removed {len(daily_sales) - len(daily_sales_clean)} outlier days")
    
    else:  # keep
        daily_sales_clean = daily_sales
    
    return daily_sales_clean

def fill_missing_dates(daily_sales):
    """Fill gaps in date sequence"""
    # Create complete date range
    full_range = pd.date_range(daily_sales.index.min(), daily_sales.index.max(), freq='D')
    
    # Reindex and forward-fill
    daily_sales_complete = daily_sales.reindex(full_range)
    missing_count = daily_sales_complete.isna().sum()
    
    if missing_count > 0:
        daily_sales_complete = daily_sales_complete.fillna(method='ffill')
        print(f"   Filled {missing_count} missing dates with forward-fill")
    
    return daily_sales_complete

def add_features(df):
    """Add useful features for modeling"""
    df_featured = df.copy()
    
    # Time features
    df_featured['year'] = df_featured.index.year
    df_featured['month'] = df_featured.index.month
    df_featured['day'] = df_featured.index.day
    df_featured['dayofweek'] = df_featured.index.dayofweek
    df_featured['is_weekend'] = df_featured['dayofweek'].isin([5, 6]).astype(int)
    df_featured['day_of_year'] = df_featured.index.dayofyear
    
    # Rolling features (lagged sales)
    df_featured['sales_lag_7'] = df_featured['sales'].shift(7)
    df_featured['sales_roll_7'] = df_featured['sales'].rolling(7).mean()
    df_featured['sales_roll_30'] = df_featured['sales'].rolling(30).mean()
    
    print(f"   Added {len(df_featured.columns) - 1} feature columns")
    
    return df_featured

def main():
    print("=" * 60)
    print("ADVANCED DATA CLEANING")
    print("=" * 60)
    
    # Load
    print("\n1. Loading data...")
    df = load_and_diagnose()
    
    # Aggregate to daily
    print("\n2. Aggregating to daily sales...")
    daily_sales = df.groupby(df['InvoiceDate'].dt.date)['TotalSales'].sum()
    print(f"   {len(daily_sales)} days")
    
    # Handle outliers
    print("\n3. Handling outliers...")
    daily_sales = handle_outliers(daily_sales, method='winsorize')
    
    # Fill missing dates
    print("\n4. Filling date gaps...")
    daily_sales = fill_missing_dates(daily_sales)
    
    # Convert to DataFrame for feature engineering
    daily_df = pd.DataFrame({'sales': daily_sales})
    
    # Add features
    print("\n5. Feature engineering...")
    daily_df = add_features(daily_df)
    
    # Save
    print("\n6. Saving cleaned data...")
    daily_df.to_csv('data/processed/daily_sales_clean.csv')
    print("   ✓ Saved to data/processed/daily_sales_clean.csv")
    
    # Summary
    print("\n" + "=" * 60)
    print("CLEANING COMPLETE")
    print("=" * 60)
    print(f"Final dataset: {len(daily_df)} days")
    print(f"Date range: {daily_df.index.min()} to {daily_df.index.max()}")
    print(f"Mean daily sales: ${daily_df['sales'].mean():,.2f}")
    print(f"Coefficient of Variation: {daily_df['sales'].std() / daily_df['sales'].mean():.2f}")
    
    return daily_df

if __name__ == "__main__":
    cleaned_df = main()