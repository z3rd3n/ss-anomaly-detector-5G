import os
import csv
from tqdm import tqdm
import json
import pandas as pd
import numpy as np

def find_raw_files(data_dir: str) -> list[str]:
    """Find all raw CSV files under data/pd_data_romes folder that are not processed yet."""
    files = []
    for root, _, filenames in os.walk(data_dir):
        for filename in filenames:
            if filename.endswith('.csv') and 'processed' not in filename:
                files.append(os.path.join(root, filename))
    return files

def safe_int(row, idx):
    try:
        return int(row[idx])
    except:
        return 0
    
def process_pdsch_row(row):
    """
    Process a PDSCH row to extract numerical features.
    No encoding needed since we'll use StandardScaler.
    """
    carrier_id = safe_int(row, 3)
    sfn = safe_int(row, 7)
    slot = safe_int(row, 8)
    harq_id = safe_int(row, 9)
    mcs = safe_int(row, 10)
    crc_status = row[15].strip().upper()
    crc = 1 if crc_status == 'PASS' else 0
    re_tx = safe_int(row, 16)

    ndi_val = 0
    ndi_part = row[-1].strip()
    if ndi_part.startswith('NDI:'):
        try:
            ndi_val = int(ndi_part.split(':')[1].strip())
        except:
            ndi_val = 0

    return {
        'SFN': sfn,
        'Slot': slot,
        #'CC': carrier_id,
        'HARQ': harq_id,
        'MCS': mcs,
        'CRC': crc,
        'ReTx': re_tx,
        'NDI': ndi_val,
    }

def main():
    data_dir = "/workspaces/thesis/data/pdsch_data_romes_clean/raw"
    output_dir = "/workspaces/thesis/data/pdsch_data_romes_clean/processed"
    os.makedirs(output_dir, exist_ok=True)

    feature_names = [
        'SFN', 'Slot', 'HARQ',
        'MCS', 'CRC', 'ReTx', 'NDI', 
    ]


    files = find_raw_files(data_dir)
    if not files:
        print("No raw files found!")
        return
    # Filter specific files
    specific_files = ['001_cc1.csv', '003_cc0.csv', '009_cc0.csv']
    files = [file for file in files if os.path.basename(file) in specific_files]
    if not files:
        print("No specific files found!")
        return

    # Second pass: encode and save processed data
    for file_name in tqdm(files):
        prev_values = {}  # Dictionary to track previous values for each path
        base_name = os.path.basename(file_name)
        name, ext = os.path.splitext(base_name)
        processed_file_name = f"{name}_processed.csv"
        processed_path = os.path.join(output_dir, processed_file_name)
        with open(processed_path, 'w', newline='') as out_f:
            writer = csv.writer(out_f)
            header = ['timestamp_str'] + feature_names
            writer.writerow(header)
            with open(file_name,'r') as in_f:
                reader = csv.reader(in_f, delimiter=';')
                in_header = next(reader,None)
                for row in reader:
                    if len(row)<3:
                        continue
                    source = row[2].strip()
                    if source == 'PDCP DL':
                        continue
                       # processed, prev_values = process_pdcp_row(row, prev_values)
                    elif source == 'PDSCH':
                        processed = process_pdsch_row(row)
                    else:
                        continue
                    timestamp_str = f"{processed_file_name};{row[1]};{processed['SFN']};{processed['Slot']}"

                    encoded_row = [timestamp_str]
                    for f in feature_names:
                        val = processed[f]
                        encoded_val = val
                        encoded_row.append(encoded_val)
                    writer.writerow(encoded_row)

    print("Data preprocessing complete. Processed files saved in data/processed_pd_data.")

if __name__ == "__main__":
    main()