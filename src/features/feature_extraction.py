"""
STEP 3: FEATURE ENGINEERING - Simple Version
"""

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def engineer_features():
    """Create features for fraud detection"""
    
    logger.info("="*70)
    logger.info("STEP 3: FEATURE ENGINEERING")
    logger.info("="*70)
    
    # Load data
    logger.info("\n📂 Loading data...")
    df = pd.read_csv('data/raw/fraud_data_raw.csv')
    logger.info(f"✅ Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    # Remove Unnamed: 0
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
        logger.info("✅ Removed 'Unnamed: 0' column")
    
    # Convert datetime
    df['TX_DATETIME'] = pd.to_datetime(df['TX_DATETIME'], dayfirst=True)
    
    # Create time features
    logger.info("\n📊 Creating features...")
    df['hour'] = df['TX_DATETIME'].dt.hour
    df['day'] = df['TX_DATETIME'].dt.day
    df['month'] = df['TX_DATETIME'].dt.month
    df['weekday'] = df['TX_DATETIME'].dt.weekday
    df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] < 6)).astype(int)
    logger.info("✅ Created time features")
    
    # Create amount features (only if columns exist)
    if 'CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW' in df.columns:
        df['amount_deviation'] = df['TX_AMOUNT'] - df['CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW']
        logger.info("✅ Created amount_deviation")
    
    if 'TX_AMOUNT' in df.columns:
        df['high_amount'] = (df['TX_AMOUNT'] > df['TX_AMOUNT'].quantile(0.95)).astype(int)
        logger.info("✅ Created high_amount")
    
    if 'CUSTOMER_ID_NB_TX_1DAY_WINDOW' in df.columns and 'CUSTOMER_ID_NB_TX_30DAY_WINDOW' in df.columns:
        df['tx_velocity'] = df['CUSTOMER_ID_NB_TX_1DAY_WINDOW'] / (df['CUSTOMER_ID_NB_TX_30DAY_WINDOW'] + 1)
        logger.info("✅ Created tx_velocity")
    
    # Drop columns
    drop_cols = ['TRANSACTION_ID', 'TX_DATETIME', 'CUSTOMER_ID', 'TERMINAL_ID']
    df = df.drop([col for col in drop_cols if col in df.columns], axis=1)
    logger.info("✅ Dropped ID columns")
    
    # Handle NaN/inf
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Save
    logger.info("\n💾 Saving...")
    df.to_csv('data/processed/features_engineered.csv', index=False)
    logger.info(f"✅ Saved: data/features_engineered.csv")
    logger.info(f"   Rows: {len(df):,}")
    logger.info(f"   Columns: {len(df.columns)}")
    
    logger.info("\n" + "="*70)
    logger.info("✅ DONE!")
    logger.info("="*70)


if __name__ == "__main__":
    try:
        engineer_features()
        print("\n🎯 Next: STEP 4 - Model Training")
    except Exception as e:
        print(f"\n❌ Error: {e}")