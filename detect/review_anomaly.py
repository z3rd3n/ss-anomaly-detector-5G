import streamlit as st
import pandas as pd
import numpy as np
import os

# ------------------------------------------------------------------------------
# 1) Utility functions
# ------------------------------------------------------------------------------

def parse_timestamp_str(timestamp_str):
    """
    Given something like: 'some_file.csv;2024-05-16 10:00:00;3;12'
    Return: file_name, raw_timestamp, sfn, slot
    """
    parts = timestamp_str.split(";")
    # Adapt if your actual format differs
    file_name    = parts[0]
    raw_ts       = parts[1]
    sfn          = int(parts[2])
    slot         = int(parts[3])
    return file_name, raw_ts, sfn, slot

# ------------------------------------------------------------------------------
# 2) Streamlit app main function
# ------------------------------------------------------------------------------

def main():
    st.title("Anomaly Labeling App")

    # User-set paths
    anomalies_csv_path = "detect/detected_anomalies.csv"
    parquet_path       = "data/scaled_pdsch.parquet"
    output_csv_path    = "labeled_anomalies.csv"
    
    # --- Load the anomalies CSV
    if not os.path.exists(anomalies_csv_path):
        st.error(f"Cannot find anomalies CSV: {anomalies_csv_path}")
        return
    
    anomalies_df = pd.read_csv(anomalies_csv_path)
    if anomalies_df.empty:
        st.warning("No rows in anomalies CSV.")
        return
    
    # --- Load the Parquet data
    if not os.path.exists(parquet_path):
        st.error(f"Cannot find Parquet file: {parquet_path}")
        return

    parquet_df = pd.read_parquet(parquet_path)
    if parquet_df.empty:
        st.warning("No rows in Parquet data.")
        return

    st.write("Loaded anomalies:", anomalies_df.shape)
    st.write("Loaded parquet:", parquet_df.shape)

    # For storing user labels in memory while app is running
    if "labels" not in st.session_state:
        # Initialize with None
        st.session_state.labels = [None] * len(anomalies_df)
    if "current_index" not in st.session_state:
        st.session_state.current_index = 0

    # If we're beyond last anomaly, let user finalize
    if st.session_state.current_index >= len(anomalies_df):
        st.success("No more anomalies to label. You can save/export your results.")
        
        # Provide a button to save
        if st.button("Save labeled anomalies to CSV"):
            labeled_df = anomalies_df.copy()
            labeled_df["user_label"] = st.session_state.labels
            labeled_df.to_csv(output_csv_path, index=False)
            st.success(f"Labeled anomalies saved to: {output_csv_path}")
        return

    # ------------------------------------------------------------------------------
    # 3) Show the current anomaly
    # ------------------------------------------------------------------------------
    idx = st.session_state.current_index
    anomaly_row = anomalies_df.iloc[idx]
    st.subheader(f"Anomaly #{idx+1} of {len(anomalies_df)}")
    st.write(anomaly_row)

    # Parse the anomaly's timestamp_str to get file_name, raw_ts, sfn, slot
    timestamp_str = anomaly_row["timestamp_str"]
    file_name, raw_ts, sfn, slot = parse_timestamp_str(timestamp_str)

    # --- Example logic to match the row inside your parquet data.
    # If your parquet has columns like file_id, SFN, Slot, etc., adapt accordingly.
    # For example, if your parquet actually has columns ["file_name", "SFN", "Slot", ...],
    # you can do:
    matched_indices = parquet_df[
        (parquet_df["SFN"] == sfn) &
        (parquet_df["Slot"] == slot)
        # If you actually store a file_name or file_id, you can do:
        # (parquet_df["file_name"] == file_name)
        # or
        # (parquet_df["file_id"] == some_file_id)
    ].index

    if len(matched_indices) == 0:
        st.warning("No exact match found in the Parquet for this anomaly.")
    else:
        # Suppose we just take the first match if multiple
        matched_idx = matched_indices[0]
        
        # --- Grab the previous 100 rows window
        window_size = 100
        start_idx = max(0, matched_idx - window_size)
        df_window = parquet_df.iloc[start_idx:matched_idx+1]
        
        st.write(f"Showing previous {window_size} rows up to index {matched_idx} in parquet:")
        st.dataframe(df_window)

    # ------------------------------------------------------------------------------
    # 4) Let user label as anomaly or not
    # ------------------------------------------------------------------------------
    user_decision = st.radio(
        "Is this truly an anomaly?",
        ("Unlabeled", "Yes", "No"),
        index=0
    )

    # ------------------------------------------------------------------------------
    # 5) Navigation & saving the temporary label
    # ------------------------------------------------------------------------------
    if st.button("Next anomaly >>"):
        # Store user label in session state
        # Could store as boolean or text
        if user_decision == "Yes":
            st.session_state.labels[idx] = True
        elif user_decision == "No":
            st.session_state.labels[idx] = False
        else:
            st.session_state.labels[idx] = None
        
        # Move to next
        st.session_state.current_index += 1
        st.experimental_rerun()

    st.write("---")
    st.write("**Progress**: ", f"{idx+1}/{len(anomalies_df)} labeled so far.")

    # Provide a button at bottom to save at any time
    if st.button("Save partial labeling so far"):
        labeled_df = anomalies_df.copy()
        labeled_df["user_label"] = st.session_state.labels
        labeled_df.to_csv(output_csv_path, index=False)
        st.success(f"Partial results saved to: {output_csv_path}")

# ------------------------------------------------------------------------------
# Streamlit entry point
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
