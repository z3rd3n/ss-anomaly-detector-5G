import csv
import os
import json
from pathlib import Path
from tqdm import tqdm
import pandas as pd

def get_unique_cc_values(input_file):
    """
    Read the CSV file and determine unique cc values present.
    Assumes there's a column indicating cc value.
    """
    df = pd.read_csv(input_file)
    # Assuming there's a column that indicates cc value
    # You'll need to replace 'cc_column' with the actual column name that contains cc values
    cc_values = sorted(df['CC'].unique())
    return cc_values

def split_csv_by_cc(input_file, output_folder, base_number):
    """
    Split a single CSV file into multiple files based on cc values.
    Shows progress bar for the splitting process.
    Returns list of output filenames.
    """
    print(f"\nReading {os.path.basename(input_file)}...")
    df = pd.read_csv(input_file)
    
    # Get unique cc values
    cc_values = get_unique_cc_values(input_file)
    output_files = []
    
    print(f"Found {len(cc_values)} unique CC values: {cc_values}")
    
    # Create progress bar for the entire process
    with tqdm(total=len(cc_values), desc="Creating CC files") as pbar:
        for cc_value in cc_values:
            # Filter data for this cc value
            cc_data = df[df['CC'] == cc_value]
            
            # Generate output filename
            output_file = os.path.join(output_folder, f'{base_number:03d}_cc{cc_value}.csv')
            
            # Save to CSV
            cc_data.to_csv(output_file, index=False)
            output_files.append(os.path.basename(output_file))
            
            pbar.update(1)
    
    return output_files

def process_folder(input_folder, output_folder, start_number=10):
    """
    Process all CSV files in the input folder and create split files in the output folder.
    Creates a mapping.json file to track file transformations.
    """
    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Get list of CSV files in input folder
    csv_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.csv')])
    total_files = len(csv_files)
    
    print(f"\nFound {total_files} CSV files to process")
    
    # Dictionary to store file mappings
    file_mapping = {}
    
    # Process each file with incremental numbering and show overall progress
    for i, csv_file in enumerate(tqdm(csv_files, desc="Overall progress", unit="file")):
        input_path = os.path.join(input_folder, csv_file)
        base_number = start_number + i
        
        # Display current file being processed
        print(f"\nProcessing file {i+1}/{total_files}: {csv_file}")
        
        # Get output filenames from split_csv
        output_files = split_csv_by_cc(input_path, output_folder, base_number)
        
        # Add to mapping dictionary
        file_mapping[csv_file] = {
            "original_file": csv_file,
            "output_files": output_files,
            "sequence_number": f"{base_number:03d}"
        }
    
    # Save mapping to JSON file
    mapping_file = os.path.join(output_folder, 'mapping.json')
    with open(mapping_file, 'w') as f:
        json.dump(file_mapping, f, indent=4)
    
    print(f"\nMapping file created: {mapping_file}")

if __name__ == '__main__':
    # Example usage
    input_folder = '/workspaces/thesis/data/processed_pdsch_data'
    output_folder = '/workspaces/thesis/data/pdsch_data_romes_clean/processed_legacy'
    start_number = 10  # Will start naming files as 010_cc0.csv, 010_cc1.csv, 010_cc2.csv, etc.
    
    print(f"Starting CSV file processing")
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    
    process_folder(input_folder, output_folder, start_number)
    
    print("\nProcessing complete!")