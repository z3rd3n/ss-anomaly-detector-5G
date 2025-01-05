import streamlit as st
import pandas as pd
import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
from utils import unscale_features

def main():
    st.set_page_config(layout="wide")  # Make the page wider
    
    # Layout: two columns, col1 (main content) and col2 (controls on the right)
    col1, col2 = st.columns([4, 1], gap="large")

    # ----------------------------------------------------------------------------
    # In col2, we place what used to be the sidebar controls
    # ----------------------------------------------------------------------------
    with col2:
        st.header("Controls")

        score_placeholder = st.empty()
        threshold_placeholder = st.empty()
        distance_placeholder = st.empty()
        
        # Window controls
        WINDOW_BEFORE = st.number_input("Rows before anomaly", min_value=0, value=20, step=1)
        WINDOW_AFTER = st.number_input("Rows after anomaly", min_value=0, value=20, step=1)

    # ----------------------------------------------------------------------------
    # Main content area (col1)
    # ----------------------------------------------------------------------------
    with col1:
        st.title("Anomaly Labeling App")

        # User-set paths (adjust if needed)
        anomalies_csv_path = "detect/detected_anomalies.csv"
        output_csv_path = "labeled_anomalies.csv"
        
        # --- Load the anomalies CSV
        if not os.path.exists(anomalies_csv_path):
            st.error(f"Cannot find anomalies CSV: {anomalies_csv_path}")
            return
        
        anomalies_df = pd.read_csv(anomalies_csv_path)
        if anomalies_df.empty:
            st.warning("No rows in anomalies CSV.")
            return
        
        st.write("Loaded anomalies:", anomalies_df.shape)

        # Initialize session state for labels
        if "labels" not in st.session_state:
            st.session_state.labels = [None] * len(anomalies_df)
        if "current_index" not in st.session_state:
            st.session_state.current_index = 0

        # If we're beyond last anomaly, let user finalize
        if st.session_state.current_index >= len(anomalies_df):
            st.success("No more anomalies to label. You can save/export your results.")
            if st.button("Save labeled anomalies to CSV"):
                labeled_df = anomalies_df.copy()
                labeled_df["user_label"] = st.session_state.labels
                labeled_df.to_csv(output_csv_path, index=False)
                st.success(f"Labeled anomalies saved to: {output_csv_path}")
            return

        # --------------------------------------------------------------------------
        # Show the current anomaly
        # --------------------------------------------------------------------------
        idx = st.session_state.current_index
        anomaly_row = anomalies_df.iloc[idx]

        total_anomalies = len(anomalies_df)

        # Extract relevant metadata
        score = anomaly_row.get("anomaly_score", "N/A")
        distance = anomaly_row.get("distance_from_threshold", "N/A")
        threshold = anomaly_row.get("threshold", "N/A")

        # Parse the anomaly's timestamp_str
        timestamp_str = anomaly_row["timestamp_str"]
        file_name = timestamp_str.split(";")[0]
        file_id = ''.join(filter(str.isdigit, file_name))  # e.g., '155' from '155.csv'

        # Sub-header with the requested format
        st.subheader(
            f"Anomaly #{idx+1} of {total_anomalies} from {file_id}.csv "
        )

        # We do NOT want to show timestamp_str, anomaly_score, distance_to_anomaly, threshold in the anomaly table
        columns_to_hide = {"timestamp_str", "anomaly_score", "distance_from_threshold", "threshold"}
        
        # Identify the columns in the current anomaly (excluding the above)
        anomaly_all_cols = anomaly_row.index.tolist()
        anomaly_display_cols = [c for c in anomaly_all_cols if c not in columns_to_hide]

        # Separate into original feature columns vs. prediction columns
        orig_cols = [c for c in anomaly_display_cols if not c.endswith("_p")]
        pred_cols = [c for c in anomaly_display_cols if c.endswith("_p")]

        # Combine original and prediction columns into a single DataFrame
        combined_df = pd.DataFrame({
            "Original Feature Columns": anomaly_row[orig_cols].values if orig_cols else [],
            "Prediction Columns": anomaly_row[pred_cols].values if pred_cols else []
        })

        # Add headers for original columns
        combined_df.index = orig_cols 

        # Show combined DataFrame
        st.write("**Original Feature Columns and Prediction Columns:**")
        st.dataframe(combined_df.T, use_container_width=True)

        # --------------------------------------------------------------------------
        # Show the window data around the matched index
        # --------------------------------------------------------------------------
        csv_file_path = os.path.join("data/scaled_pdsch_data", file_id + "_scaled.csv")
        
        if not os.path.exists(csv_file_path):
            st.error(f"No matching CSV file found for file_id: {file_id}")
            return
        
        df = pd.read_csv(csv_file_path)
        matched_indices = df[df['timestamp_str'] == timestamp_str].index

        if len(matched_indices) == 0:
            st.warning("No exact match found in the CSV for this anomaly.")
        else:
            matched_idx = matched_indices[0]
            start_idx = max(0, matched_idx - WINDOW_BEFORE)
            end_idx = min(len(df), matched_idx + WINDOW_AFTER + 1)

            # Create the window around the anomaly, removing timestamp_str from display
            df_window = df.iloc[start_idx:end_idx].drop(columns=["timestamp_str"], errors="ignore")
            df_window = unscale_features(df_window)

            # Highlight the row of interest in the window
            def highlight_anomaly_row(row):
                return [
                    'background-color: yellow' if row.name == matched_idx else ''
                    for _ in row
                ]
            
            st.write(
                f"Showing {WINDOW_BEFORE} rows before and {WINDOW_AFTER} rows after index {matched_idx} in CSV:"
            )
            styled_window = df_window.style.apply(highlight_anomaly_row, axis=1)
            st.dataframe(styled_window, use_container_width=True)

        st.write("---")

    # ----------------------------------------------------------------------------
    # Decision & labeling controls (side-by-side buttons) in col2
    # ----------------------------------------------------------------------------
    with col2:
        score_placeholder.markdown(f"<h3>Score: {score:.4f}</h3>", unsafe_allow_html=True)
        threshold_placeholder.markdown(f"<h3>Threshold: {threshold:.4f}</h3>", unsafe_allow_html=True)
        distance_placeholder.markdown(f"<h3>Diff: {distance:.4f}</h3>", unsafe_allow_html=True)

        # Two side-by-side buttons: "Anomaly" or "Normal"
        bcol1, bcol2 = st.columns(2)
        with bcol1:
            if st.button("Anomaly"):
                st.session_state.labels[idx] = True
                # Save immediately
                labeled_df = anomalies_df.copy()
                labeled_df["user_label"] = st.session_state.labels
                labeled_df.to_csv(output_csv_path, index=False)
                # Move to next anomaly
                st.session_state.current_index += 1
                st.rerun()

        with bcol2:
            if st.button("Normal"):
                st.session_state.labels[idx] = False
                # Save immediately
                labeled_df = anomalies_df.copy()
                labeled_df["user_label"] = st.session_state.labels
                labeled_df.to_csv(output_csv_path, index=False)
                # Move to next anomaly
                st.session_state.current_index += 1
                st.rerun()

        # Progress status
        st.write("**Progress**: ", f"{idx+1}/{len(anomalies_df)} labeled so far.")

# --------------------------------------------------------------------------
# Streamlit entry point
# --------------------------------------------------------------------------
if __name__ == "__main__":
    main()
