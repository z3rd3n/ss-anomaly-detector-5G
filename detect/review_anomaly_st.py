import os
import sys
import json

import streamlit as st
import pandas as pd
import pyarrow.parquet as pq

# If needed, adjust your app's base directory for imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils import unscale_features

@st.cache_data
def load_file_mapping(mapping_path: str = "file_mapping.json"):
    """
    Reads the file_mapping.json that maps something like {"155": 0, "160": 1, ...}.
    """
    with open(mapping_path, "r") as f:
        return json.load(f)

@st.cache_data
def load_parquet_for_file(numeric_id_val: int, parquet_path: str = "data/scaled_pdsch.parquet"):
    """
    Loads only the rows matching file_id == numeric_id_val from the Parquet file.
    This is much more efficient than reading the entire file.
    """
    if not os.path.exists(parquet_path):
        return pd.DataFrame()  # or None

    # Filter to get only rows for the given numeric_id
    df = pd.read_parquet(
        parquet_path,
        filters=[("file_id", "=", numeric_id_val)]
    )
    return df

def parse_original_id(timestamp_str: str) -> str:
    """
    Extracts the file portion before the semicolon, e.g. "155.csv" from "155.csv; foo"
    Then returns just the digits ("155") from "155.csv".
    """
    file_part = timestamp_str.split(";")[0]  # e.g. "155.csv"
    return "".join(filter(str.isdigit, file_part))  # "155"

def main():
    st.set_page_config(layout="wide")
    
    # 1. Load the anomalies CSV
    anomalies_csv_path = "detect/detected_anomalies.csv"
    if not os.path.exists(anomalies_csv_path):
        st.error(f"Cannot find anomalies CSV: {anomalies_csv_path}")
        return
    
    anomalies_df = pd.read_csv(anomalies_csv_path)
    if anomalies_df.empty:
        st.warning("No rows in anomalies CSV.")
        return
    
    st.title("Anomaly Labeling App")
    st.write("Loaded anomalies:", anomalies_df.shape)

    # 2. Load the file mapping JSON
    mapping_dict = load_file_mapping("data/file_mapping.json")

    # 3. Add columns for original_file_id (e.g. "155") and numeric_id
    anomalies_df["original_file_id"] = anomalies_df["timestamp_str"].apply(parse_original_id)
    anomalies_df["numeric_id"] = anomalies_df["original_file_id"].apply(lambda x: mapping_dict.get(x, None))

    # Drop anomalies that don't match a numeric_id in the mapping
    anomalies_df = anomalies_df.dropna(subset=["numeric_id"]).copy()
    anomalies_df["numeric_id"] = anomalies_df["numeric_id"].astype(int)

    if anomalies_df.empty:
        st.warning("All anomalies filtered out because no matching numeric_id was found.")
        return

    # 4. Group anomalies by numeric_id
    grouped = anomalies_df.groupby("numeric_id")
    # Each group -> df_of_anomalies_for_that_file
    anomaly_groups = {nid: g.reset_index(drop=True) for nid, g in grouped}

    # Sort the file IDs so we process them in ascending numeric_id order
    file_ids_in_order = sorted(anomaly_groups.keys())

    # Create an overall structure for storing labels
    # We'll store them in "user_label" column in anomaly_groups
    if "labeled_groups" not in st.session_state:
        # A dict { numeric_id: pd.DataFrame of anomalies with user_label col } 
        st.session_state.labeled_groups = anomaly_groups

    # Keep track of which file index we are on
    if "file_idx" not in st.session_state:
        st.session_state.file_idx = 0  # which index in file_ids_in_order
    # Keep track of anomaly index within the current file
    if "anomaly_idx" not in st.session_state:
        st.session_state.anomaly_idx = 0

    # If we've processed all files, let user finalize
    if st.session_state.file_idx >= len(file_ids_in_order):
        st.success("No more anomalies to label! All files have been processed.")

        # Let user save labeled anomalies to CSV (optional final save)
        if st.button("Save labeled anomalies to CSV"):
            final_dfs = []
            for fid in st.session_state.labeled_groups:
                final_dfs.append(st.session_state.labeled_groups[fid])
            labeled_all = pd.concat(final_dfs, ignore_index=True)
            labeled_all.to_csv("labeled_anomalies.csv", index=False)
            st.success("Labeled anomalies saved to labeled_anomalies.csv.")
        return

    # Otherwise, get the current file's anomaly DataFrame
    current_file_id = file_ids_in_order[st.session_state.file_idx]
    current_df = st.session_state.labeled_groups[current_file_id]

    # If we've gone past the last anomaly for this file, move to the next file
    if st.session_state.anomaly_idx >= len(current_df):
        st.session_state.file_idx += 1
        st.session_state.anomaly_idx = 0
        st.rerun()
        return

    # 5. Load the data for the current file, once
    file_data = load_parquet_for_file(current_file_id)
    if file_data is None or file_data.empty:
        st.error(f"No data found for numeric_id={current_file_id} in Parquet.")
        return

    # Provide controls for how many rows before/after we show
    col1, col2 = st.columns([4, 1], gap="large")
    with col2:
        st.header("Controls")
        scores = st.empty()
        WINDOW_BEFORE = st.number_input("Rows before anomaly", min_value=0, value=20, step=1)
        WINDOW_AFTER = st.number_input("Rows after anomaly", min_value=0, value=20, step=1)

    # 6. Show the current anomaly
    with col1:
        anomaly_idx = st.session_state.anomaly_idx
        anomaly_row = current_df.iloc[anomaly_idx]

        # find the key whose value is current_file_id
        key = [k for k, v in mapping_dict.items() if v == current_file_id][0]
        st.subheader(f"File {key}.csv (file_id={current_file_id}), Anomaly {anomaly_idx+1} / {len(current_df)}")

        score = anomaly_row.get("anomaly_score", "N/A")
        distance = anomaly_row.get("distance_from_threshold", "N/A")
        threshold = anomaly_row.get("threshold", "N/A")
        timestamp_str = anomaly_row["timestamp_str"]

        # Show other features (excluding certain columns)
        columns_to_hide = {
            "timestamp_str", "anomaly_score", "distance_from_threshold", "threshold",
            "original_file_id", "numeric_id", "user_label"
        }
        anomaly_display_cols = [c for c in anomaly_row.index if c not in columns_to_hide]

        # Separate original vs. prediction columns
        orig_cols = [c for c in anomaly_display_cols if not c.endswith("_p")]
        pred_cols = [c for c in anomaly_display_cols if c.endswith("_p")]

        if orig_cols or pred_cols:
            st.write("**Original Feature Columns and Prediction Columns:**")
            combined_df = pd.DataFrame({
                "Original Feature Columns": anomaly_row[orig_cols].values if orig_cols else [],
                "Prediction Columns": anomaly_row[pred_cols].values if pred_cols else []
            })
            # Label the rows using orig_cols as an index
            combined_df.index = orig_cols if orig_cols else pred_cols
            st.dataframe(combined_df.T, use_container_width=True)

        # 7. Locate this anomaly in the file_data
        matched_rows = file_data[file_data["timestamp_str"] == timestamp_str]
        if len(matched_rows) == 0:
            st.warning(f"No match for timestamp_str = {timestamp_str} in file numeric_id={current_file_id}.")
        else:
            matched_idx = matched_rows.index[0]
            start_idx = max(0, matched_idx - WINDOW_BEFORE)
            end_idx = min(len(file_data), matched_idx + WINDOW_AFTER + 1)

            # Unscale the window
            df_window = file_data.iloc[start_idx:end_idx].copy()
            # drop the timestamp_str column and file_id column
            df_window = df_window.drop(columns=["timestamp_str", "file_id"])
            df_window = unscale_features(df_window)

            # Highlight the anomaly row
            def highlight_anomaly_row(row):
                return [
                    "background-color: yellow" if row.name == matched_idx else ""
                    for _ in row
                ]

            st.markdown(
                f"Showing {WINDOW_BEFORE} rows before and {WINDOW_AFTER} rows after index {matched_idx}."
            )
            styled_window = df_window.style.apply(highlight_anomaly_row, axis=1)
            st.dataframe(styled_window, use_container_width=True)

    # --- HELPER FUNCTION TO SAVE LABELED ANOMALIES ---
    def save_labeled_anomalies():
        """
        Gather all labeled groups from session state and save
        to labeled_anomalies.csv.
        """
        all_dfs = []
        for fid in st.session_state.labeled_groups:
            all_dfs.append(st.session_state.labeled_groups[fid])
        labeled_all = pd.concat(all_dfs, ignore_index=True)
        labeled_all.to_csv("labeled_anomalies.csv", index=False)

    # 8. Labeling buttons
    with col2:
        if isinstance(score, float) and isinstance(threshold, float) and isinstance(distance, float):
            scores.write(
                f"<h5>Score: {score:.4f}</h5>"
                f"<h5>Threshold: {threshold:.4f}</h5>"
                f"<h5>Diff: {distance:.4f}</h5>",
                unsafe_allow_html=True
            )
        else:
            scores.write(
                f"<h5>Score: {score}</h5>"
                f"<h5>Threshold: {threshold}</h5>"
                f"<h5>Diff: {distance}</h5>",
                unsafe_allow_html=True
            )

        st.write("---")
        anomaly_button = st.button("Anomaly")
        normal_button = st.button("Normal")

        if anomaly_button:
            current_df.loc[anomaly_idx, "user_label"] = True
            st.session_state.anomaly_idx += 1
            
            # Save immediately after labeling
            st.session_state.labeled_groups[current_file_id] = current_df
            save_labeled_anomalies()
            
            st.rerun()

        if normal_button:
            current_df.loc[anomaly_idx, "user_label"] = False
            st.session_state.anomaly_idx += 1
            
            # Save immediately after labeling
            st.session_state.labeled_groups[current_file_id] = current_df
            save_labeled_anomalies()
            
            st.rerun()

    # 9. Store the updated DataFrame back in session
    st.session_state.labeled_groups[current_file_id] = current_df

if __name__ == "__main__":
    main()
