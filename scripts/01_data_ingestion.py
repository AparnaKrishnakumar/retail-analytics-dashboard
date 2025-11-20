"""
Data Ingestion Script
Loads raw retail data and performs initial validation
"""

import pandas as pd
import os
from datetime import datetime

def create_directories():
    """Ensure required directories exist"""
    dirs = ['data/raw', 'data/processed', 'data/output']
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    print("✓ Directories created")

def load_raw_data(filepath):
    """Load raw retail data from CSV"""
    try:
        df = pd.read_csv(filepath, encoding='ISO-8859-1')
        print(f"✓ Loaded {len(df):,} records")
        print(f"✓ Columns: {list(df.columns)}")
        return df
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        print("Please download dataset and place in data/raw/")
        return None

def validate_data(df):
    """Perform basic data quality checks"""
    print("\n=== Data Quality Report ===")
    print(f"Shape: {df.shape}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nDate range: {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")
    return df

def basic_cleaning(df):
    """Initial data cleaning"""
    initial_rows = len(df)
    
    # Remove rows with missing CustomerID (if present)
    if 'CustomerID' in df.columns:
        df = df.dropna(subset=['CustomerID'])
    
    # Remove cancelled orders (InvoiceNo starting with 'C')
    if 'InvoiceNo' in df.columns:
        df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
    
    # Remove negative quantities
    if 'Quantity' in df.columns:
        df = df[df['Quantity'] > 0]
    
    # Remove negative prices
    if 'UnitPrice' in df.columns:
        df = df[df['UnitPrice'] > 0]
    
    print(f"\n✓ Cleaned: {initial_rows:,} → {len(df):,} rows")
    return df

def save_processed_data(df, output_path):
    """Save cleaned data"""
    df.to_csv(output_path, index=False)
    print(f"✓ Saved to: {output_path}")

def main():
    """Run data ingestion pipeline"""
    print(" Starting Data Ingestion Pipeline\n")
    
    # Setup
    create_directories()
    
    # Load
    df = load_raw_data('data/raw/OnlineRetail.csv')
    if df is None:
        return
    
    # Validate
    df = validate_data(df)
    
    # Clean
    df = basic_cleaning(df)
    
    # Save
    save_processed_data(df, 'data/processed/clean_data.csv')
    
    print("\n Data ingestion complete!")
    print(f" Ready for analysis: {len(df):,} records")

if __name__ == "__main__":
    main()