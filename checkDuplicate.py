import pyarrow.parquet as pq
import pandas as pd
from tqdm import tqdm
from collections import Counter

# Path to the Parquet file
parquet_path = "data/scaled_pdsch.parquet"

# Read Parquet metadata without reading the entire file
parquet_file = pq.ParquetFile(parquet_path)

# A counter to hold timestamp_str -> count
timestamp_counter = Counter()

# Iterate over each row group in the Parquet file
for rg_index in tqdm(range(parquet_file.num_row_groups), desc="Processing row groups"):
    # Read only the columns needed
    table = parquet_file.read_row_group(rg_index, columns=["timestamp_str"])
    
    # Convert to Pandas DataFrame
    df_chunk = table.to_pandas()
    
    # Update the counter with all timestamps in this chunk
    timestamp_counter.update(df_chunk["timestamp_str"])

# Now determine which timestamps are duplicated
duplicates = [(ts, cnt) for ts, cnt in timestamp_counter.items() if cnt > 1]

if duplicates:
    # Create a DataFrame with duplicates only
    duplicates_df = pd.DataFrame(duplicates, columns=["timestamp_str", "count"])
    duplicates_df.to_csv("duplicates.csv", index=False)
    print(f"Found {len(duplicates)} duplicated timestamps. Written to duplicates.csv")
else:
    print("No duplicates found.")
