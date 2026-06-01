import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def preprocess_data(train_path="data/application_train.csv", bureau_path="data/bureau.csv", output_path="data/processed_train.csv"):
    """
    Production-ready data preprocessing pipeline.
    Loads datasets, aggregates credit bureau features, merges, imputes missing values, and encodes categories.
    """
    print(f"Loading train dataset: {train_path}...")
    df_train = pd.read_csv(train_path)
    
    print(f"Loading bureau dataset: {bureau_path}...")
    df_bureau = pd.read_csv(bureau_path)
    
    # 2. Aggregate bureau.csv by SK_ID_CURR
    print("Aggregating credit bureau records...")
    
    # Pre-allocate aggregation DataFrame for speed and clarity
    bureau_agg = pd.DataFrame(index=df_bureau['SK_ID_CURR'].unique())
    bureau_agg.index.name = 'SK_ID_CURR'
    
    # total_previous_loans
    bureau_agg['total_previous_loans'] = df_bureau.groupby('SK_ID_CURR').size()
    
    # active_credit_count
    bureau_agg['active_credit_count'] = df_bureau[df_bureau['CREDIT_ACTIVE'] == 'Active'].groupby('SK_ID_CURR').size()
    
    # closed_credit_count
    bureau_agg['closed_credit_count'] = df_bureau[df_bureau['CREDIT_ACTIVE'] == 'Closed'].groupby('SK_ID_CURR').size()
    
    # average_days_credit
    bureau_agg['average_days_credit'] = df_bureau.groupby('SK_ID_CURR')['DAYS_CREDIT'].mean()
    
    # average_credit_enddate
    bureau_agg['average_credit_enddate'] = df_bureau.groupby('SK_ID_CURR')['DAYS_CREDIT_ENDDATE'].mean()
    
    # average_credit_update
    bureau_agg['average_credit_update'] = df_bureau.groupby('SK_ID_CURR')['DAYS_CREDIT_UPDATE'].mean()
    
    bureau_agg = bureau_agg.reset_index()
    
    # Count columns engineered
    engineered_cols = ['total_previous_loans', 'active_credit_count', 'closed_credit_count', 
                       'average_days_credit', 'average_credit_enddate', 'average_credit_update']
    
    # 3. Merge aggregated bureau features into application_train.csv
    print("Merging aggregated bureau features into train dataset...")
    df_merged = df_train.merge(bureau_agg, on='SK_ID_CURR', how='left')
    
    # Fill count NaNs with 0 explicitly (since missing records imply 0 counts)
    count_cols = ['total_previous_loans', 'active_credit_count', 'closed_credit_count']
    for col in count_cols:
        df_merged[col] = df_merged[col].fillna(0)
        
    # 4. Handle missing values
    print("Handling missing values...")
    
    # Exclude SK_ID_CURR and TARGET from imputation/encoding logic
    exclude_cols = ['SK_ID_CURR', 'TARGET']
    
    # Identify numerical columns (excluding keys)
    num_cols = df_merged.select_dtypes(include=[np.number]).columns.difference(exclude_cols)
    # Identify categorical columns (excluding keys)
    cat_cols = df_merged.select_dtypes(include=['object', 'category']).columns.difference(exclude_cols)
    
    # Numeric columns -> median imputation
    print(f"Imputing {len(num_cols)} numerical columns with median...")
    for col in num_cols:
        median_val = df_merged[col].median()
        # In case the whole column is NaN (safety fallback)
        if pd.isna(median_val):
            median_val = 0.0
        df_merged[col] = df_merged[col].fillna(median_val)
        
    # Categorical columns -> most frequent value (mode)
    print(f"Imputing {len(cat_cols)} categorical columns with most frequent value...")
    for col in cat_cols:
        mode_series = df_merged[col].mode()
        if not mode_series.empty:
            mode_val = mode_series[0]
        else:
            mode_val = "Unknown"
        df_merged[col] = df_merged[col].fillna(mode_val)
        
    # 5. Encode categorical columns
    print(f"Encoding {len(cat_cols)} categorical columns using LabelEncoder...")
    for col in cat_cols:
        le = LabelEncoder()
        df_merged[col] = le.fit_transform(df_merged[col].astype(str))
        
    # 6. Preserve TARGET column
    # (Confirmed: TARGET has been preserved and excluded from imputation/encoding)
    
    # Ensure parent output dir exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 7. Save final dataset
    print(f"Saving processed dataset to {output_path}...")
    df_merged.to_csv(output_path, index=False)
    print("Save completed successfully.")
    
    # 8. Print stats
    print("\n" + "="*50)
    print("PREPROCESSING PIPELINE EXECUTION SUMMARY")
    print("="*50)
    print(f"Final dataset shape: {df_merged.shape[0]:,} rows, {df_merged.shape[1]} columns")
    print(f"Number of engineered features: {len(engineered_cols)}")
    
    total_missing_after = df_merged.isnull().sum().sum()
    print(f"Total missing values after preprocessing: {total_missing_after}")
    print("="*50 + "\n")

if __name__ == "__main__":
    preprocess_data()
