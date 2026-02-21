"""
STEP 2.5: DATA VISUALIZATION - Fraud Detection (Updated)
Create visualizations with ALL features including Terminal features
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os

# Simple logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


def create_visualizations():
    """Create comprehensive visualizations for fraud detection data"""
    
    logger.info("="*70)
    logger.info("STEP 2.5: DATA VISUALIZATION (UPDATED)")
    logger.info("="*70)
    
    try:
        # Create output folder
        os.makedirs('reports/figures', exist_ok=True)
        logger.info("\n📁 Created reports/figures folder")
        
        # Load data
        logger.info("\n📂 Loading data...")
        df = pd.read_csv('data/raw/fraud_data_raw.csv')
        logger.info(f"✅ Loaded {len(df):,} rows, {len(df.columns)} columns")
        
        # Remove Unnamed: 0 if exists
        if 'Unnamed: 0' in df.columns:
            df = df.drop('Unnamed: 0', axis=1)
        
        # Convert datetime
        if 'TX_DATETIME' in df.columns:
            df['TX_DATETIME'] = pd.to_datetime(df['TX_DATETIME'], dayfirst=True)
        
        # ========================================
        # 1. FRAUD DISTRIBUTION
        # ========================================
        logger.info("\n📊 Creating Plot 1: Fraud Distribution...")
        
        plt.figure(figsize=(10, 6))
        
        fraud_counts = df['TX_FRAUD'].value_counts()
        colors = ['#2ecc71', '#e74c3c']
        labels = ['Legitimate', 'Fraudulent']
        
        plt.pie(fraud_counts, labels=labels, autopct='%1.2f%%', 
                colors=colors, startangle=90, textprops={'fontsize': 12})
        plt.title('Transaction Distribution: Fraud vs Legitimate', 
                 fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig('reports/figures/01_fraud_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("✅ Saved: 01_fraud_distribution.png")
        
        # ========================================
        # 2. AMOUNT DISTRIBUTION
        # ========================================
        logger.info("\n📊 Creating Plot 2: Amount Distribution...")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Overall distribution
        axes[0].hist(df['TX_AMOUNT'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Transaction Amount ($)', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Overall Transaction Amount Distribution', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # By fraud status
        df[df['TX_FRAUD']==0]['TX_AMOUNT'].hist(bins=50, alpha=0.6, label='Legitimate', 
                                                  color='green', ax=axes[1])
        df[df['TX_FRAUD']==1]['TX_AMOUNT'].hist(bins=50, alpha=0.6, label='Fraudulent', 
                                                  color='red', ax=axes[1])
        axes[1].set_xlabel('Transaction Amount ($)', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title('Transaction Amount: Fraud vs Legitimate', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('reports/figures/02_amount_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("✅ Saved: 02_amount_distribution.png")
        
        # ========================================
        # 3. FULL CORRELATION HEATMAP
        # ========================================
        logger.info("\n📊 Creating Plot 3: Full Correlation Heatmap...")
        
        # Select ALL numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove ID and time columns
        cols_to_exclude = ['TRANSACTION_ID', 'CUSTOMER_ID', 'TERMINAL_ID', 'TX_DATETIME', 
                          'TX_TIME_SECONDS', 'TX_TIME_DAYS', 'Unnamed: 0']
        numerical_cols = [col for col in numerical_cols if col not in cols_to_exclude]
        
        logger.info(f"   Including {len(numerical_cols)} numerical features")
        
        # Calculate correlation
        correlation = df[numerical_cols].corr()
        
        # Create large heatmap
        plt.figure(figsize=(20, 18))
        
        sns.heatmap(correlation, 
                   annot=False,  # Too many to annotate
                   cmap='RdYlGn',  # Red-Yellow-Green
                   center=0, 
                   square=True, 
                   linewidths=0.5,
                   cbar_kws={"shrink": 0.8, "label": "Correlation"},
                   xticklabels=True,
                   yticklabels=True)
        
        plt.title('Feature Correlation Heatmap - All Numeric Features', 
                 fontsize=18, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        plt.savefig('reports/figures/03_correlation_heatmap_full.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("✅ Saved: 03_correlation_heatmap_full.png")
        
        # ========================================
        # 4. CORRELATION WITH TX_FRAUD
        # ========================================
        logger.info("\n📊 Creating Plot 4: Correlation with TX_FRAUD...")
        
        # Get correlation with target
        fraud_corr = correlation['TX_FRAUD'].drop('TX_FRAUD').sort_values(ascending=False)
        
        # Get top 20 positive and negative
        top_features = pd.concat([fraud_corr.head(20), fraud_corr.tail(20)])
        
        # Create color map
        colors = ['red' if x > 0 else 'blue' for x in top_features.values]
        
        plt.figure(figsize=(12, 12))
        plt.barh(range(len(top_features)), top_features.values, color=colors, alpha=0.7)
        plt.yticks(range(len(top_features)), top_features.index, fontsize=9)
        plt.xlabel('Correlation with TX_FRAUD', fontsize=12, fontweight='bold')
        plt.title('Top Features Correlated with Fraud', fontsize=14, fontweight='bold')
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        plt.grid(True, alpha=0.3, axis='x')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='red', alpha=0.7, label='Positive (more fraud)'),
                          Patch(facecolor='blue', alpha=0.7, label='Negative (less fraud)')]
        plt.legend(handles=legend_elements, loc='best')
        
        plt.tight_layout()
        plt.savefig('reports/figures/04_fraud_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("✅ Saved: 04_fraud_correlation.png")
        
        # ========================================
        # 5. TERMINAL RISK ANALYSIS
        # ========================================
        logger.info("\n📊 Creating Plot 5: Terminal Risk Analysis...")
        
        terminal_risk_cols = [col for col in df.columns if 'TERMINAL_ID_RISK' in col]
        
        if terminal_risk_cols:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            
            for idx, col in enumerate(terminal_risk_cols[:3]):
                df[df['TX_FRAUD']==0][col].hist(bins=30, alpha=0.6, label='Legitimate', 
                                                 color='green', ax=axes[idx])
                df[df['TX_FRAUD']==1][col].hist(bins=30, alpha=0.6, label='Fraudulent', 
                                                 color='red', ax=axes[idx])
                axes[idx].set_xlabel('Risk Score', fontsize=11)
                axes[idx].set_ylabel('Frequency', fontsize=11)
                axes[idx].set_title(col.replace('TERMINAL_ID_', ''), fontsize=12, fontweight='bold')
                axes[idx].legend()
                axes[idx].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('reports/figures/05_terminal_risk.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("✅ Saved: 05_terminal_risk.png")
        
        # ========================================
        # 6. CUSTOMER VS TERMINAL FEATURES
        # ========================================
        logger.info("\n📊 Creating Plot 6: Customer vs Terminal Features...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Customer 1-day window
        if 'CUSTOMER_ID_NB_TX_1DAY_WINDOW' in df.columns:
            df.boxplot(column='CUSTOMER_ID_NB_TX_1DAY_WINDOW', by='TX_FRAUD', ax=axes[0,0])
            axes[0,0].set_title('Customer: 1-Day Transaction Count')
            axes[0,0].set_xlabel('TX_FRAUD')
            axes[0,0].set_ylabel('Count')
        
        # Customer 7-day avg amount
        if 'CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW' in df.columns:
            df.boxplot(column='CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW', by='TX_FRAUD', ax=axes[0,1])
            axes[0,1].set_title('Customer: 7-Day Avg Amount')
            axes[0,1].set_xlabel('TX_FRAUD')
            axes[0,1].set_ylabel('Amount ($)')
        
        # Terminal 1-day window
        if 'TERMINAL_ID_NB_TX_1DAY_WINDOW' in df.columns:
            df.boxplot(column='TERMINAL_ID_NB_TX_1DAY_WINDOW', by='TX_FRAUD', ax=axes[1,0])
            axes[1,0].set_title('Terminal: 1-Day Transaction Count')
            axes[1,0].set_xlabel('TX_FRAUD')
            axes[1,0].set_ylabel('Count')
        
        # Terminal 7-day risk
        if 'TERMINAL_ID_RISK_7DAY_WINDOW' in df.columns:
            df.boxplot(column='TERMINAL_ID_RISK_7DAY_WINDOW', by='TX_FRAUD', ax=axes[1,1])
            axes[1,1].set_title('Terminal: 7-Day Risk Score')
            axes[1,1].set_xlabel('TX_FRAUD')
            axes[1,1].set_ylabel('Risk Score')
        
        plt.suptitle('Customer vs Terminal Features by Fraud Status', 
                    fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig('reports/figures/06_customer_terminal_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("✅ Saved: 06_customer_terminal_comparison.png")
        
        # ========================================
        # 7. FEATURE IMPORTANCE PREVIEW
        # ========================================
        logger.info("\n📊 Creating Plot 7: Feature Statistics...")
        
        # Compare means for fraud vs legitimate
        fraud_data = df[df['TX_FRAUD']==1][numerical_cols].mean()
        legit_data = df[df['TX_FRAUD']==0][numerical_cols].mean()
        
        # Get top 15 by difference
        diff = abs(fraud_data - legit_data).sort_values(ascending=False).head(15)
        top_cols = diff.index
        
        comparison = pd.DataFrame({
            'Fraudulent': fraud_data[top_cols],
            'Legitimate': legit_data[top_cols]
        })
        
        plt.figure(figsize=(12, 10))
        comparison.plot(kind='barh', color=['red', 'green'], alpha=0.7)
        plt.xlabel('Average Value', fontsize=12, fontweight='bold')
        plt.ylabel('Features', fontsize=12)
        plt.title('Top 15 Features: Fraud vs Legitimate (Mean Values)', 
                 fontsize=14, fontweight='bold')
        plt.legend(['Fraudulent', 'Legitimate'])
        plt.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig('reports/figures/07_feature_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("✅ Saved: 07_feature_comparison.png")
        
        # ========================================
        # SUMMARY
        # ========================================
        logger.info("\n" + "="*70)
        logger.info("📊 VISUALIZATION SUMMARY")
        logger.info("="*70)
        logger.info("\nCreated visualizations:")
        logger.info("  1. ✅ Fraud Distribution")
        logger.info("  2. ✅ Amount Distribution")
        logger.info("  3. ✅ Full Correlation Heatmap (ALL features)")
        logger.info("  4. ✅ Correlation with TX_FRAUD")
        logger.info("  5. ✅ Terminal Risk Analysis")
        logger.info("  6. ✅ Customer vs Terminal Comparison")
        logger.info("  7. ✅ Feature Comparison")
        
        logger.info(f"\n📋 Features analyzed: {len(numerical_cols)}")
        logger.info(f"   Including Customer features (1/7/30-day windows)")
        logger.info(f"   Including Terminal features (1/7/30-day windows + risks)")
        
        logger.info("\n📁 All plots saved in: reports/figures/")
        
        logger.info("\n" + "="*70)
        logger.info("✅ STEP 2.5: DATA VISUALIZATION COMPLETE")
        logger.info("="*70)
        
        return True
        
    except FileNotFoundError as e:
        logger.error(f"\n❌ Error: File not found - {e}")
        logger.error("💡 Run STEP 1 first")
        raise
        
    except Exception as e:
        logger.error(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    try:
        create_visualizations()
        print(f"\n🎯 Next: Ask for STEP 3 - Feature Engineering")
        print(f"\n💡 View your plots in: reports/figures/")
        
    except Exception as e:
        print(f"\n💥 Failed: {str(e)}")