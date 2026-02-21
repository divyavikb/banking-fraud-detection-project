"""
STEP 2: DATA VALIDATION - Fraud Detection
Check data quality and identify issues
"""

import pandas as pd
import numpy as np
import logging
import os

# Simple logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)
os.makedirs("reports", exist_ok=True)

def validate_data():
    """Validate fraud detection data quality"""
    
    logger.info("="*70)
    logger.info("STEP 2: DATA VALIDATION")
    logger.info("="*70)
    
    try:
        # Load data from Step 1
        logger.info("\n📂 Loading data from Step 1...")
        df = pd.read_csv('data/raw/fraud_data_raw.csv')
        logger.info(f"✅ Loaded {len(df):,} rows")
        
        # === 1. CHECK MISSING VALUES ===
        logger.info("\n" + "="*70)
        logger.info("1️⃣ CHECKING MISSING VALUES")
        logger.info("="*70)
        
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        
        has_missing = False
        for col in df.columns:
            if missing[col] > 0:
                logger.warning(f"⚠️  {col}: {missing[col]:,} missing ({missing_pct[col]:.2f}%)")
                has_missing = True
        
        if not has_missing:
            logger.info("✅ No missing values found!")
        
        # === 2. CHECK DUPLICATES ===
        logger.info("\n" + "="*70)
        logger.info("2️⃣ CHECKING DUPLICATE TRANSACTIONS")
        logger.info("="*70)
        
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            logger.warning(f"⚠️  Found {duplicates:,} duplicate rows ({duplicates/len(df)*100:.2f}%)")
            logger.info("💡 Action: Will remove duplicates in Step 3")
        else:
            logger.info("✅ No duplicates found!")
        
        # Check duplicate transaction IDs
        if 'TRANSACTION_ID' in df.columns:
            dup_ids = df['TRANSACTION_ID'].duplicated().sum()
            if dup_ids > 0:
                logger.warning(f"⚠️  Found {dup_ids:,} duplicate transaction IDs")
            else:
                logger.info("✅ All transaction IDs are unique")
        
        # === 3. CHECK DATA TYPES ===
        logger.info("\n" + "="*70)
        logger.info("3️⃣ CHECKING DATA TYPES")
        logger.info("="*70)
        
        logger.info("\nCurrent data types:")
        for col, dtype in df.dtypes.items():
            logger.info(f"  {col}: {dtype}")
        
        # Check if TX_DATETIME needs conversion
        if 'TX_DATETIME' in df.columns:
            if df['TX_DATETIME'].dtype == 'object':
                logger.warning("⚠️  TX_DATETIME is object (string) - should be datetime")
                logger.info("💡 Action: Will convert to datetime in Step 3")
            else:
                logger.info("✅ TX_DATETIME already in correct format")
        
        # === 4. CHECK VALUE RANGES ===
        logger.info("\n" + "="*70)
        logger.info("4️⃣ CHECKING VALUE RANGES")
        logger.info("="*70)
        
        # Check transaction amounts
        if 'TX_AMOUNT' in df.columns:
            min_amount = df['TX_AMOUNT'].min()
            max_amount = df['TX_AMOUNT'].max()
            
            logger.info(f"\n💰 Transaction Amounts:")
            logger.info(f"  Min: ${min_amount:.2f}")
            logger.info(f"  Max: ${max_amount:.2f}")
            
            if min_amount < 0:
                logger.warning(f"⚠️  Found negative amounts! Min: ${min_amount:.2f}")
            else:
                logger.info("✅ All amounts are positive")
            
            if max_amount > 100000:
                logger.warning(f"⚠️  Very high amounts detected! Max: ${max_amount:.2f}")
                logger.info("💡 Check: Are these legitimate or data errors?")
        
        # Check fraud labels
        if 'TX_FRAUD' in df.columns:
            unique_values = df['TX_FRAUD'].unique()
            logger.info(f"\n🚨 Fraud Labels: {sorted(unique_values)}")
            
            if not set(unique_values).issubset({0, 1}):
                logger.warning(f"⚠️  Fraud labels should be 0 or 1, found: {unique_values}")
            else:
                logger.info("✅ Fraud labels are valid (0 and 1)")
        
        # === 5. CHECK CLASS IMBALANCE ===
        logger.info("\n" + "="*70)
        logger.info("5️⃣ CHECKING CLASS IMBALANCE")
        logger.info("="*70)
        
        if 'TX_FRAUD' in df.columns:
            fraud_count = df['TX_FRAUD'].sum()
            legit_count = len(df) - fraud_count
            fraud_pct = (fraud_count / len(df)) * 100
            
            logger.info(f"\nClass Distribution:")
            logger.info(f"  Legitimate (0): {legit_count:,} ({100-fraud_pct:.2f}%)")
            logger.info(f"  Fraudulent (1): {fraud_count:,} ({fraud_pct:.2f}%)")
            logger.info(f"  Imbalance Ratio: 1:{int(legit_count/fraud_count)}")
            
            if fraud_pct < 5:
                logger.warning("⚠️  Highly imbalanced dataset!")
                logger.info("💡 Action: Use SMOTE or class weights in Step 3")
            else:
                logger.info("✅ Dataset is reasonably balanced")
        
        # === 6. CHECK OUTLIERS ===
        logger.info("\n" + "="*70)
        logger.info("6️⃣ CHECKING FOR OUTLIERS")
        logger.info("="*70)
        
        if 'TX_AMOUNT' in df.columns:
            Q1 = df['TX_AMOUNT'].quantile(0.25)
            Q3 = df['TX_AMOUNT'].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df['TX_AMOUNT'] < lower_bound) | (df['TX_AMOUNT'] > upper_bound)]
            outlier_count = len(outliers)
            outlier_pct = (outlier_count / len(df)) * 100
            
            logger.info(f"\nOutlier Detection (IQR method):")
            logger.info(f"  Q1: ${Q1:.2f}")
            logger.info(f"  Q3: ${Q3:.2f}")
            logger.info(f"  IQR: ${IQR:.2f}")
            logger.info(f"  Lower bound: ${lower_bound:.2f}")
            logger.info(f"  Upper bound: ${upper_bound:.2f}")
            logger.info(f"  Outliers found: {outlier_count:,} ({outlier_pct:.2f}%)")
            
            if outlier_pct > 10:
                logger.warning("⚠️  High percentage of outliers detected")
                logger.info("💡 Note: Some outliers might be fraud cases - don't remove blindly!")
        
        # === 7. VALIDATION SUMMARY ===
        logger.info("\n" + "="*70)
        logger.info("7️⃣ VALIDATION SUMMARY")
        logger.info("="*70)
        
        issues_found = []
        
        if has_missing:
            issues_found.append("Missing values")
        if duplicates > 0:
            issues_found.append("Duplicate rows")
        if 'TX_DATETIME' in df.columns and df['TX_DATETIME'].dtype == 'object':
            issues_found.append("DateTime format")
        if fraud_pct < 5:
            issues_found.append("Class imbalance")
        
        if issues_found:
            logger.warning(f"\n⚠️  Issues to fix in Step 3:")
            for issue in issues_found:
                logger.warning(f"   - {issue}")
        else:
            logger.info("\n✅ No major issues found!")
        
        # Save validation report
        validation_report = {
        'total_rows': int(len(df)),
        'total_columns': int(len(df.columns)),
        'missing_values': bool(has_missing),
        'duplicates': int(duplicates),
        'fraud_rate': float(fraud_pct),
        'outliers': int(outlier_count),
        'issues': issues_found
}
        
        # Save report
        import json
        with open('reports/validation_report.json', 'w') as f:
            json.dump(validation_report, f, indent=2)
        
        logger.info(f"\n📄 Validation report saved: data/validation_report.json")
        
        logger.info("\n" + "="*70)
        logger.info("✅ STEP 2: DATA VALIDATION COMPLETE")
        logger.info("="*70)
        
        return validation_report
        
    except FileNotFoundError:
        logger.error("\n❌ Error: data/raw/fraud_data_raw.csv not found")
        logger.error("💡 Run STEP 1 first: python src/data/data_ingestion.py")
        raise
        
    except Exception as e:
        logger.error(f"\n❌ Error: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        report = validate_data()
        print(f"\n🎯 Next: Ask for STEP 3 - Data Transformation")
        
    except Exception as e:
        print(f"\n💥 Failed: {str(e)}")