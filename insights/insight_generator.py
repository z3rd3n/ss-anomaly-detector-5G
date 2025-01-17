import sqlite3
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Avoid plotting to screen
import matplotlib.pyplot as plt
from tqdm import tqdm

# ======================================================
# Configuration (adjust if needed)
# ======================================================
TABLE_NAME = "anomalies_table"
MAX_RETX = 4  # from your other project

# Create the table if it doesn't exist
CREATE_TABLE_QUERY = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    id INTEGER PRIMARY KEY,
    timestamp_str TEXT,
    SFN INTEGER,
    Slot INTEGER,
    CC INTEGER,
    HARQ INTEGER,
    MCS INTEGER,
    CRC INTEGER,
    ReTx INTEGER,
    NDI INTEGER
);
"""

# ======================================================
# 1) Helper: Check if table already exists
# ======================================================
def table_exists(db_path, table_name):
    """Return True if the given table_name exists in the DB, else False."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        SELECT name
        FROM sqlite_master
        WHERE type='table' AND name=?;
    """, (table_name,))
    result = c.fetchone()
    conn.close()
    return (result is not None)

# ======================================================
# 2) Load CSV into DB
#    (Always re-creates the table here by dropping any old one)
# ======================================================
def load_csv_into_db(csv_path, db_path, table_name):
    """
    Reads a CSV file and inserts into SQLite. 
    Overwrites the old table (if it existed).
    """
    # If DB already exists and table exists, drop it
    if os.path.exists(db_path) and table_exists(db_path, table_name):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(f"DROP TABLE IF EXISTS {table_name};")
        conn.commit()
        conn.close()

    # Create DB (if needed) and table, then insert the data
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(CREATE_TABLE_QUERY)
    conn.commit()

    # Read the CSV with pandas
    df = pd.read_csv(csv_path)
    # Keep only relevant columns
    needed_cols = ["timestamp_str", "SFN", "Slot", "CC", "HARQ", "MCS", "CRC", "ReTx", "NDI"]
    existing_cols = [c for c in needed_cols if c in df.columns]
    df = df[existing_cols].copy()

    # Convert to numeric where appropriate
    for col in ["SFN", "Slot", "CC", "HARQ", "MCS", "CRC", "ReTx", "NDI"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    insert_query = f"""
    INSERT INTO {table_name} 
    (timestamp_str, SFN, Slot, CC, HARQ, MCS, CRC, ReTx, NDI)
    VALUES (?,?,?,?,?,?,?,?,?)
    """

    # Insert row by row
    for _, row in df.iterrows():
        data_tuple = (
            row.get("timestamp_str", ""),
            row.get("SFN", 0),
            row.get("Slot", 0),
            row.get("CC", 0),
            row.get("HARQ", 0),
            row.get("MCS", 0),
            row.get("CRC", 0),
            row.get("ReTx", 0),
            row.get("NDI", 0),
        )
        cursor.execute(insert_query, data_tuple)

    conn.commit()
    conn.close()

# ======================================================
# 3) Generate Rule-Based Insights
#    (Same logic, just reading from DB)
# ======================================================
def generate_insights(db_path, table_name):
    conn = sqlite3.connect(db_path)

    # Dictionary: row_id -> list of triggered rules
    row_insights = {}

    def add_insight(row_id, message):
        if row_id not in row_insights:
            row_insights[row_id] = []
        row_insights[row_id].append(message)

    # --- "unnecessary_retx" ---
    query_unnecessary_retx = f"""
    WITH windowed AS (
        SELECT
            id,
            ReTx,
            CRC,
            LAG(CRC) OVER (PARTITION BY CC, HARQ ORDER BY id) AS prev_crc
        FROM {table_name}
    )
    SELECT id
    FROM windowed
    WHERE ReTx > 0 AND prev_crc = 1;
    """

    # --- "missing_retx" ---
    query_missing_retx = f"""
    WITH windowed AS (
        SELECT
            id,
            ReTx,
            CRC,
            LAG(CRC) OVER (PARTITION BY CC, HARQ ORDER BY id) AS prev_crc
        FROM {table_name}
    )
    SELECT id
    FROM windowed
    WHERE ReTx = 0 AND prev_crc = 0;
    """

    # --- "new_data_no_retx" ---
    query_new_data_no_retx = f"""
    WITH windowed AS (
        SELECT
            id,
            CRC,
            NDI,
            LAG(CRC) OVER (PARTITION BY CC, HARQ ORDER BY id) AS prev_crc,
            LAG(NDI) OVER (PARTITION BY CC, HARQ ORDER BY id) AS prev_ndi
        FROM {table_name}
    )
    SELECT id
    FROM windowed
    WHERE prev_crc = 0
      AND NDI <> prev_ndi;
    """

    # --- "max_retx_achieved" ---
    query_max_retx_achieved = f"""
    SELECT id
    FROM {table_name}
    WHERE ReTx >= {MAX_RETX};
    """

    # Execute queries, collect row IDs
    for (row_id,) in conn.execute(query_unnecessary_retx):
        add_insight(row_id, "unnecessary_retx")

    for (row_id,) in conn.execute(query_missing_retx):
        add_insight(row_id, "missing_retx")

    for (row_id,) in conn.execute(query_new_data_no_retx):
        add_insight(row_id, "new_data_no_retx")

    for (row_id,) in conn.execute(query_max_retx_achieved):
        add_insight(row_id, "max_retx_achieved")

    conn.close()
    return row_insights

# ======================================================
# 4) Merge insights back into a CSV (only lines that have insights)
# ======================================================
def merge_insights_and_write_csv(db_path, table_name, row_insights, csv_output):
    """
    Read the table from DB, attach 'insight' column, 
    then write only the rows that have at least one rule triggered.
    """
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM {table_name};", conn)
    conn.close()

    # Create 'insight' column (comma-separated if multiple)
    def insight_list_to_str(row_id):
        return ", ".join(row_insights[row_id]) if row_id in row_insights else ""

    df["insight"] = df["id"].apply(insight_list_to_str)

    # Keep only rows that have at least one insight
    df_insights = df[df["insight"] != ""].copy()

    # If no rows triggered any rule, we can still write out an empty file 
    # (or skip). Here, we write it (empty).
    df_insights.to_csv(csv_output, index=False)
    print(f"[INFO] Wrote {csv_output} ({len(df_insights)} rows with rule-based insights).")

# ======================================================
# main() driver
# ======================================================
def main():
    input_folder = "/workspaces/thesis/data/processed_pdsch_data/"
    output_folder = "/workspaces/thesis/data/processed_pdsch_insights/"
    os.makedirs(output_folder, exist_ok=True)
    
    # List all CSV files in the input folder
    all_files = [f for f in os.listdir(input_folder) if f.endswith(".csv")]
    
    # Progress bar for files
    for csv_file in tqdm(all_files, desc="Processing CSV files"):
        # Build paths
        csv_input_path = os.path.join(input_folder, csv_file)
        csv_output_name = os.path.splitext(csv_file)[0] + "_insights.csv"
        csv_output_path = os.path.join(output_folder, csv_output_name)
        
        # Temporary DB path (can be the same name each loop, we'll delete after)
        db_path = "/workspaces/thesis/temp_mlflow/anomalies_rules.db"
        
        # 1) Load CSV into DB
        load_csv_into_db(csv_input_path, db_path, TABLE_NAME)
        
        # 2) Generate rule-based insights
        row_insights = generate_insights(db_path, TABLE_NAME)
        
        # 3) Merge insights into new CSV (only lines that triggered a rule)
        merge_insights_and_write_csv(db_path, TABLE_NAME, row_insights, csv_output_path)
        
        # 4) Delete the .db file
        if os.path.exists(db_path):
            os.remove(db_path)
            # Uncomment if you want a confirmation:
            # print(f"[INFO] Deleted DB file: {db_path}")

if __name__ == "__main__":
    main()
