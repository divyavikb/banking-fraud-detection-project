"""
STEP 1: DATA INGESTION - Fraud Detection
"""

import os
import pandas as pd
from pathlib import Path
import logging

# Simple logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# CHANGE THIS LINE!
YOUR_DATA_FILE = "D:\\banking_fraud_dataset.csv"
os.makedirs("data/raw", exist_ok=True)

def main():
    """Load and save fraud data"""
    
    logger.info("="*70)
    logger.info("STEP 1: DATA INGESTION")
    logger.info("="*70)
    
    try:
                
        # Load data
        logger.info(f"\n📂 Loading: {YOUR_DATA_FILE}")
        df = pd.read_csv(YOUR_DATA_FILE)
        logger.info(f"✅ Loaded {len(df):,} rows, {len(df.columns)} columns")
        
        # Show columns
        logger.info(f"\n📋 Columns: {list(df.columns)}")
        
        # Fraud stats
        if 'TX_FRAUD' in df.columns:
            fraud_count = df['TX_FRAUD'].sum()
            fraud_pct = (fraud_count / len(df)) * 100
            logger.info(f"\n🚨 Fraud: {fraud_pct:.2f}% ({fraud_count:,} fraudulent)")
        
        # Save
        output_file = "data/raw/fraud_data_raw.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"\n💾 Saved to: {output_file}")
        
        logger.info("\n" + "="*70)
        logger.info("✅ SUCCESS! STEP 1 COMPLETE")
        logger.info("="*70)
        
        return output_file
        
    except FileNotFoundError:
        logger.error(f"\n❌ File not found: {YOUR_DATA_FILE}")
        raise
        
    except Exception as e:
        logger.error(f"\n❌ Error: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        output_file = main()
        print(f"\n🎯 Next: Ask for STEP 2")
    except Exception as e:
        print(f"\n💥 Failed: {str(e)}")
