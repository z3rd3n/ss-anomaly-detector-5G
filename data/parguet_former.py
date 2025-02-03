import os
import json
import glob
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# Directory containing your scaled CSVs
scaled_dir = "/workspaces/thesis/data/pdsch_data_romes_clean/processed_validation"
parquet_file = "/workspaces/thesis/data/pdsch_data_romes_clean/processed_validation/val_unscaled_pdsch.parquet"
mapping_json = "/workspaces/thesis/data/pdsch_data_romes_clean/processed_validation/val_unscaled_file_mapping.json"

# Gather all scaled CSV files
processed_files = sorted(glob.glob(os.path.join(scaled_dir, "*_processed.csv")))

# Initialize Parquet writer
writer = None

# Initialize a dictionary to store { original_file_id: numeric_id }
file_mapping_dict = {}

for i, fpath in enumerate(tqdm(processed_files, desc="Processing files")):
    df = pd.read_csv(fpath)
    # drop Source column
    if 'CC' in df.columns:
        df = df.drop(columns=['CC'])
    if 'Source' in df.columns:
        df= df.drop(columns=['Source'])

    # Example file path: "data/scaled_pdsch_data/155_scaled.csv"
    base_name = os.path.basename(fpath)            # "155_scaled.csv"
    original_file_id = base_name.replace("_processed.csv", "")  # "155"

    # Store in dictionary
    file_mapping_dict[original_file_id] = i

    # Add numeric_id to the dataframe
    df['file_id'] = i

    # Convert to Parquet Table and write
    table = pa.Table.from_pandas(df)
    if writer is None:
        writer = pq.ParquetWriter(
            parquet_file,
            table.schema,
            compression='snappy'
        )
    writer.write_table(table)

# Close the Parquet writer
if writer is not None:
    writer.close()

#Save the dictionary to a JSON file
with open(mapping_json, "w") as f:
    json.dump(file_mapping_dict, f, indent=2)

print(f"Done. Parquet saved to {parquet_file}, mapping saved to {mapping_json}.")
