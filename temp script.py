"""
Comprehensive Exploratory Data Analysis (EDA) for UCI Mushroom Dataset
Author: Data Science Analysis
Date: October 2025
"""
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency
import warnings

warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# ============================================================================
# 1. DATA LOADING AND INITIAL EXPLORATION
# ============================================================================

def load_mushroom_data():
  """"""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"

    # Column names based on UCI documentation
    columns = [
        'class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
        'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
        'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
        'stalk-surface-below-ring', 'stalk-color-above-ring',
        'stalk-color-below-ring', 'veil-type', 'veil-color',
        'ring-number', 'ring-type', 'spore-print-color',
        'population', 'habitat'
    ]

    # Load data
    df = pd.read_csv(url, names=columns)

    # Replace '?' with NaN
    df = df.replace('?', np.nan)

    return df


def initial_data_exploration(df):
    """"""
    print("=" * 80)
    print("MUSHROOM DATASET - EXPLORATORY DATA ANALYSIS")
    print("=" * 80)
    print("\n1. DATASET OVERVIEW")
    print("-" * 80)
    print(f"Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\nFirst 5 rows:")
    print(df.head())

    print("\n\n2. DATA TYPES")
    print("-" * 80)
    print(df.dtypes)

    print("\n\n3. DATASET INFO")
    print("-" * 80)
    df.info()

    print("\n\n4. BASIC STATISTICS")
    print("-" * 80)
    print(df.describe(include='all'))

    print("\n\n5. MISSING VALUES")
    print("-" * 80)
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Percentage': missing_pct
    })
    print(missing_df[missing_df['Missing Count'] > 0])

    return df

"""
# ============================================================================
# 2. TARGET VARIABLE ANALYSIS
# ============================================================================


# ============================================================================
# 3. UNIVARIATE ANALYSIS
# ============================================================================



# ============================================================================
# 4. BIVARIATE ANALYSIS
# ============================================================================



# ============================================================================
# 5. CORRELATION ANALYSIS
# ============================================================================

def correlation_analysis(df):
    """Analyze correlations between features"""
    print("\n\n" + "=" * 80)
    print("CORRELATION ANALYSIS")
    print("=" * 80)

    # Label encode all features for correlation
    df_encoded = df.copy()
    for col in df_encoded.columns:
        df_encoded[col] = pd.Categorical(df_encoded[col]).codes

    # Calculate correlation matrix
    corr_matrix = df_encoded.corr()

    # Visualize correlation matrix
    plt.figure(figsize=(20, 16))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                cmap='coolwarm', center=0, square=True,
                linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
    print("\n✓ Correlation matrix saved as 'correlation_matrix.png'")
    plt.show()

    # Find highly correlated features (excluding diagonal)
    print("\nHighly Correlated Feature Pairs (|correlation| > 0.5):")
    print("-" * 80)
    high_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.5:
                high_corr.append({
                    'Feature 1': corr_matrix.columns[i],
                    'Feature 2': corr_matrix.columns[j],
                    'Correlation': corr_matrix.iloc[i, j]
                })

    if high_corr:
        high_corr_df = pd.DataFrame(high_corr)
        high_corr_df = high_corr_df.sort_values('Correlation', ascending=False, key=abs)
        print(high_corr_df.to_string(index=False))
    else:
        print("No highly correlated feature pairs found.")


# ============================================================================
# 6. ADVANCED INSIGHTS
# ============================================================================

def generate_insights(df, chi_df):
    """Generate key insights and recommendations"""
    print("\n\n" + "=" * 80)
    print("KEY INSIGHTS AND RECOMMENDATIONS")
    print("=" * 80)

    # 1. Class balance
    class_balance = df['class'].value_counts()
    balance_ratio = min(class_balance) / max(class_balance)

    print("\n1. CLASS BALANCE")
    print("-" * 80)
    print(f"Balance ratio: {balance_ratio:.3f}")
    if balance_ratio > 0.8:
        print("✓ Dataset is well-balanced. No resampling needed.")
    else:
        print("⚠ Dataset is imbalanced. Consider SMOTE or class weights.")

    # 2. Most predictive features
    print("\n2. MOST PREDICTIVE FEATURES")
    print("-" * 80)
    top_5 = chi_df.head(5)
    for idx, row in top_5.iterrows():
        print(f"• {row['Feature']}: Cramér's V = {row['Cramers V']:.3f}")

    # 3. Odor analysis (critical feature)
    print("\n3. ODOR ANALYSIS (MOST CRITICAL FEATURE)")
    print("-" * 80)
    odor_class = pd.crosstab(df['odor'], df['class'])
    print(odor_class)

    # Find perfect predictors
    print("\nPerfect Predictors:")
    for odor_type in odor_class.index:
        edible = odor_class.loc[odor_type, 'e']
        poisonous = odor_class.loc[odor_type, 'p']
        if edible == 0:
            print(f"  • Odor '{odor_type}': 100% POISONOUS")
        elif poisonous == 0:
            print(f"  • Odor '{odor_type}': 100% EDIBLE")

    # 4. Missing values impact
    print("\n4. MISSING VALUES")
    print("-" * 80)
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print("Features with missing values:")
        for col, count in missing[missing > 0].items():
            pct = (count / len(df)) * 100
            print(f"  • {col}: {count} ({pct:.2f}%)")
            print(
                f"    Recommendation: {'Impute with mode' if pct < 5 else 'Consider dropping or advanced imputation'}")
    else:
        print("✓ No missing values detected.")

    # 5. Constant features
    print("\n5. CONSTANT/LOW-VARIANCE FEATURES")
    print("-" * 80)
    for col in df.columns:
        if col != 'class':
            unique = df[col].nunique()
            if unique == 1:
                print(f"  • {col}: Only 1 unique value - REMOVE THIS FEATURE")
            elif unique == 2:
                value_counts = df[col].value_counts()
                ratio = value_counts.iloc[0] / len(df)
                if ratio > 0.95:
                    print(f"  • {col}: Very low variance ({ratio:.1%} dominant) - Consider removing")

    # 6. Modeling recommendations
    print("\n6. MODELING RECOMMENDATIONS")
    print("-" * 80)
    print("Preprocessing:")
    print("  ✓ Label encode all categorical features")
    print("  ✓ Handle missing values in 'stalk-root'")
    print("  ✓ Remove constant features (e.g., veil-type)")
    print("  ✓ Feature selection: Use top 10-15 features based on Cramér's V")
    print("\nAlgorithm Selection:")
    print("  ✓ Decision Trees (interpretable, handles categorical well)")
    print("  ✓ Random Forest (ensemble, robust)")
    print("  ✓ XGBoost/CatBoost (high performance, native categorical support)")
    print("  ✓ Logistic Regression (baseline model)")
    print("\nValidation Strategy:")
    print("  ✓ Stratified K-Fold Cross-Validation (k=5 or 10)")
    print("  ✓ Hold-out test set (20-30%)")
    print("  ✓ Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC")


