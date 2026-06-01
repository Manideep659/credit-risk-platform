import os
import sqlite3
import pandas as pd

def load_data_to_sqlite(csv_path="data/processed_train.csv", db_path="data/credit_risk.db", table_name="loan_data"):
    """
    Production-grade SQLite loader.
    Loads processed CSV data, writes it to a SQLite table, verifies metrics,
    and constructs optimization indexes on TARGET, CODE_GENDER, and DAYS_BIRTH.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Source preprocessed dataset not found at: {csv_path}")
        
    # Ensure database parent directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    print(f"Loading preprocessed dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {df.shape[0]:,} rows and {df.shape[1]} columns in-memory.")
    
    print(f"Connecting to SQLite database: {db_path}...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        print(f"Writing dataset to SQLite table '{table_name}' (this may take a few moments)...")
        # Load all rows using pandas to_sql chunked for memory efficiency and speed
        df.to_sql(table_name, conn, if_exists="replace", index=False, chunksize=10000)
        print("Data ingestion complete.")
        
        # 4. Verify Row and Column Counts
        print("Verifying database table integrity...")
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        db_rows = cursor.fetchone()[0]
        
        cursor.execute(f"PRAGMA table_info({table_name})")
        db_cols = len(cursor.fetchall())
        
        # Verify match with original DataFrame
        if db_rows == df.shape[0] and db_cols == df.shape[1]:
            print(f"[VERIFIED] Table matches original shape perfectly.")
        else:
            print(f"[WARNING] Shape mismatch! DB: ({db_rows}, {db_cols}), DF: {df.shape}")
            
        # 5. Create indexes on TARGET, CODE_GENDER, DAYS_BIRTH for SQL querying speed
        print("Creating indexes on search fields (TARGET, CODE_GENDER, DAYS_BIRTH)...")
        
        cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_target ON {table_name} (TARGET)")
        cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_gender ON {table_name} (CODE_GENDER)")
        cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_birth ON {table_name} (DAYS_BIRTH)")
        
        # Commit index additions
        conn.commit()
        print("Index creation complete.")
        
        # Verify Indexes
        cursor.execute(f"PRAGMA index_list({table_name})")
        indexes = cursor.fetchall()
        print(f"Active indexes on table '{table_name}':")
        for idx in indexes:
            print(f" - {idx[1]} (Unique: {idx[2]})")
            
        # Summary Printout
        print("\n" + "="*50)
        print("SQLITE INGESTION SUMMARY")
        print("="*50)
        print(f"Database File: {os.path.abspath(db_path)}")
        print(f"Table Name:    {table_name}")
        print(f"Row Count:     {db_rows:,}")
        print(f"Column Count:  {db_cols}")
        print(f"Indexes Built: idx_target, idx_gender, idx_birth")
        print("="*50 + "\n")
        
    except Exception as e:
        conn.rollback()
        print(f"[ERROR] Database ingestion failed: {e}")
        raise e
    finally:
        conn.close()
        print("SQLite connection closed.")

if __name__ == "__main__":
    load_data_to_sqlite()
