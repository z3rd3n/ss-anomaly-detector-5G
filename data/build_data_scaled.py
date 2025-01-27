import os
import csv
from tqdm import tqdm
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib  # for saving the scaler

def safe_int(row, idx):
    try:
        return int(row[idx])
    except:
        return 0
    
def find_raw_files(data_dir: str) -> list[str]:
    """Find all raw CSV files under data/pd_data_romes folder that are not processed yet."""
    files = []
    for root, _, filenames in os.walk(data_dir):
        for filename in filenames:
            if filename.endswith('.csv') and 'processed' not in filename:
                files.append(os.path.join(root, filename))
    return files

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
    data_dir = "/workspaces/thesis/data/pdsch_data_romes_clean/processed"
    output_dir = "/workspaces/thesis/data/pdsch_data_romes_clean/scaled"
    os.makedirs(output_dir, exist_ok=True)

    feature_names = [
        'SFN', 'Slot', 'HARQ', 'MCS', 'CRC', 'ReTx', 'NDI']

    # First pass: collect all data for scaling
    print("Collecting data for scaling...")
    all_data = []
    all_timestamps = []
    
    files = find_raw_files(data_dir)
    if not files:
        print("No raw files found!")
        return

    for file_name in tqdm(files):
        with open(file_name, 'r') as f:
            reader = csv.reader(f, delimiter=';')
            header = next(reader, None)
            for row in reader:
                if len(row) < 3:
                    continue
                source = row[2].strip()
                if source == 'PDSCH':
                    processed = process_pdsch_row(row)
                    timestamp_str = f"{os.path.basename(file_name)};{row[1]};{processed['SFN']};{processed['Slot']}"
                    feature_values = [processed[f] for f in feature_names]
                    all_data.append(feature_values)
                    all_timestamps.append(timestamp_str)

    # Convert to numpy array for scaling
    X = np.array(all_data)
    
    # Initialize and fit StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save the scaler
    scaler_path = os.path.join(output_dir, 'standard_scaler.pkl')
    joblib.dump(scaler, scaler_path)
    
    # Save scaling parameters for reference
    scaling_params = {
        'mean_': scaler.mean_.tolist(),
        'scale_': scaler.scale_.tolist(),
        'var_': scaler.var_.tolist(),
        'feature_names': feature_names
    }
    with open(os.path.join(output_dir, 'scaling_params.json'), 'w') as f:
        json.dump(scaling_params, f, indent=2)

    # Save scaled data
    print("Saving scaled data...")
    for file_idx, file_name in enumerate(files):
        base_name = os.path.basename(file_name)
        name, ext = os.path.splitext(base_name)
        scaled_file_name = f"{name}_scaled.csv"
        scaled_path = os.path.join(output_dir, scaled_file_name)
        
        with open(scaled_path, 'w', newline='') as out_f:
            writer = csv.writer(out_f)
            header = ['timestamp_str'] + feature_names
            writer.writerow(header)
            
            # Write scaled data
            mask = [timestamp.startswith(base_name) for timestamp in all_timestamps]
            file_data = X_scaled[mask]
            file_timestamps = np.array(all_timestamps)[mask]
            
            for timestamp, scaled_row in zip(file_timestamps, file_data):
                writer.writerow([timestamp] + scaled_row.tolist())

    # Test scaling/unscaling
    print("\nTesting scaling/unscaling...")
    # Load saved scaler
    loaded_scaler = joblib.load(scaler_path)
    
    # Take a sample of original data
    sample_idx = np.random.randint(0, len(X), 5)
    original_sample = X[sample_idx]
    scaled_sample = scaler.transform(original_sample)
    unscaled_sample = loaded_scaler.inverse_transform(scaled_sample)
    
    print("\nScaling Test Results:")
    print("Original Values:")
    print(original_sample)
    print("\nScaled Values:")
    print(scaled_sample)
    print("\nUnscaled Values:")
    print(unscaled_sample)
    print("\nDifference (original - unscaled):")
    print(np.abs(original_sample - unscaled_sample).max())
    
    print("\nData preprocessing complete. Files saved in:", output_dir)
    print("Scaler saved as:", scaler_path)

if __name__ == "__main__":
    main()