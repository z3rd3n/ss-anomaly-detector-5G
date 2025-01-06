import os
import logging
import streamlit as st
import pandas as pd
import torch
import mlflow
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import sys
import json
from mlflow.tracking import MlflowClient

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from detect.mlflow_utils import load_config_from_mlflow, download_artifact
from utils import load_checkpoint, plot_attention_matrices, unscale_features
from approaches.subAdjacent.trainer import detect_anomalies
from data.dataLoader import ParquetSequenceDataset, custom_collate_fn

@st.cache_data
def load_file_mapping(mapping_path: str = "data/file_mapping.json"):
    """
    Reads the file_mapping.json that maps something like {"155": 0, "160": 1, ...}.
    Adjust the path as needed.
    """
    with open(mapping_path, "r") as f:
        return json.load(f)

@st.cache_data
def load_parquet_for_file(numeric_id_val: int, parquet_path: str = "data/scaled_pdsch.parquet"):
    """
    Loads only the rows matching file_id == numeric_id_val from the Parquet file.
    This is more efficient than reading the entire file if it is partitioned 
    or if you use 'filters' in read_parquet.
    Adjust the path as needed.
    """
    if not os.path.exists(parquet_path):
        return pd.DataFrame()

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

def app_main():
    # Increase logging level
    logging.basicConfig(level=logging.INFO)
    st.set_page_config(layout="wide", page_title="Anomaly Labeling + MLflow Demo")

    # For demonstration, let's have multiple pages in Streamlit using st.tabs
    tabs = st.tabs([
        "MLflow Model Loader", 
        "Anomaly Detection & Labeling", 
        "Attention Matrices", 
        "Highest Score Anomalies"
    ])
    
    # ------------------- TAB 0: MLflow Model Loader ---------------------------
    with tabs[0]:
        st.header("MLflow Model Loader")
        st.write("This section downloads the model from MLflow and sets up the environment.")
        tracking_uri = st.text_input(
            "MLflow Tracking URI (e.g. http://localhost:8000)",
            value="http://localhost:8000",
            help="Must be http(s) if using mlflow-artifacts store"
        )
        run_id = st.text_input("Enter MLflow run_id:", value="595c7686c17a4926b81ab7175b44e63b", help="Paste the run_id from MLflow")
        if st.button("Load Model & Config from MLflow"):
            if not run_id:
                st.error("Please provide a run_id.")
            
            else:
                with st.spinner("Downloading and loading model..."):
                    mlflow.set_tracking_uri(tracking_uri)
                    st.session_state.temp_dir = "temp_mlflow"
                    os.makedirs(st.session_state.temp_dir, exist_ok=True)

                    # 1) Download and import the config_class
                    ConfigClass = load_config_from_mlflow(run_id, st.session_state.temp_dir)
                    st.session_state.params = ConfigClass()
                    st.session_state.params.output_dir = st.session_state.temp_dir
                    # 3) Download checkpoint
                    ckpt_path = download_artifact(run_id, "checkpoints/checkpoint_best.pt", st.session_state.temp_dir)
                    mlflow.set_tracking_uri(tracking_uri)
                    # If the run is known to be in subAdjacent, do this
                    mlflow.set_experiment("subAdjacent")
                    
                    # Only do this once:
                    if mlflow.active_run() is None:
                        mlflow.start_run(run_id=run_id)
                    # 4) Build model from the config
                    st.session_state.model = st.session_state.params.build_model()
                    # Build optimizer
                    if hasattr(st.session_state.params, "optimizer") and st.session_state.params.optimizer == "AdamW":
                        st.session_state.optimizer = torch.optim.AdamW(
                            st.session_state.model.parameters(),
                            lr=st.session_state.params.learning_rate,
                            weight_decay=st.session_state.params.weight_decay
                        )
                    else:
                        st.session_state.optimizer = torch.optim.Adam(
                            st.session_state.model.parameters(),
                            lr=st.session_state.params.learning_rate
                        )
                    # 5) Load checkpoint
                    load_checkpoint(st.session_state.model, st.session_state.optimizer, ckpt_path)
                    
                    
                    st.success("Model & config loaded from MLflow. Params updated.")
                    st.write("Current parameters:")
                    st.json({key: value for key, value in vars(st.session_state.params).items()})

    # ------------------- TAB 1: Anomaly Detection & Labeling ------------------
    with tabs[1]:
        st.header("Anomaly Detection & Labeling")
        st.write("This section runs anomaly detection, saves results, and allows labeling anomalies grouped by file/folder.")

        if "model" not in st.session_state:
            st.warning("Please go to the 'MLflow Model Loader' tab to load a model first.")
        else:
            # Let user specify p and q
            st.subheader("Run Anomaly Detection with custom p and q")
            p_val = st.number_input("Percentile (p)", min_value=0, max_value=100, value=95, step=1)
            q_val = st.number_input("Quartile (q)", min_value=0.0, max_value=1.0, value=0.99, step=0.01)
            st.session_state.params.q = p_val
            st.session_state.params.p = q_val

            _, val_dataset = ParquetSequenceDataset.create_train_val_splits(
                parquet_path=st.session_state.params.parquet_path,
                feature_columns=st.session_state.params.feature_columns,
                seq_len=st.session_state.params.seq_len,
                validation_ratio=st.session_state.params.validation_ratio,
                seed=st.session_state.params.seed
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=st.session_state.params.batch_size,
                num_workers=st.session_state.params.num_workers,
                collate_fn=custom_collate_fn,
                pin_memory=st.session_state.params.pin_memory,
                drop_last=True
            )

            st.session_state.val_loader = val_loader

            if st.button("Run Anomaly Detection"):
                with st.spinner("Running detection..."):
                    logging.info("Creating train and validation datasets...")

                    if os.path.exists("temp_mlflow/detected_anomalies.csv"):
                        st.warning("Anomalies CSV already exists. Overwriting.")
                        os.remove("temp_mlflow/detected_anomalies.csv")

                    if os.path.exists("temp_mlflow/labeled_anomalies.csv"):
                        st.warning("Labeled anomalies CSV already exists. Overwriting.")
                        os.remove("temp_mlflow/labeled_anomalies.csv") 

                    anomalies_df, fig_path = detect_anomalies(
                        params=st.session_state.params, 
                        model=st.session_state.model, 
                        val_loader=val_loader
                    )

                    # Add user_label column
                    anomalies_df["user_label"] = None

                    # Save to CSV
                    output_csv = "temp_mlflow/detected_anomalies.csv"
                    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
                    anomalies_df.to_csv(output_csv, index=False)
                    st.success(f"Anomalies saved to {output_csv}")

                    # Log final q, p, plus number of anomalies found
                    mlflow.log_param("final_q", q_val)
                    mlflow.log_param("final_p", p_val)
                    mlflow.log_metric("num_anomalies", len(anomalies_df))

                    # Also log the anomalies CSV to MLflow
                    mlflow.log_artifact(output_csv, artifact_path="detection_results", run_id=run_id)

                    # Log the anomaly scores figure if it exists
                    if fig_path and os.path.exists(fig_path):
                        mlflow.log_artifact(fig_path, artifact_path="plots", run_id=run_id)

                    # Keep anomalies in session_state
                    st.session_state.detected_anomalies_df = anomalies_df

            st.write("---")
            # ------  Labeling UI  ------
            anomalies_csv_path = "temp_mlflow/detected_anomalies.csv"
            if not os.path.exists(anomalies_csv_path):
                st.warning("No anomalies CSV found. Please run anomaly detection first.")
            else:
                anomalies_df = pd.read_csv(anomalies_csv_path)
                if "user_label" not in anomalies_df.columns:
                    anomalies_df["user_label"] = None
                if anomalies_df.empty:
                    st.warning("No rows in anomalies CSV.")
                else:
                    st.write(f"Loaded {len(anomalies_df)} anomalies from {anomalies_csv_path} for labeling.")
                    
                    # Merge with existing labeled anomalies if it exists
                    labeled_csv_path = "temp_mlflow/labeled_anomalies.csv"
                    if os.path.exists(labeled_csv_path):
                        labeled_previous = pd.read_csv(labeled_csv_path)
                        if "user_label" in labeled_previous.columns:
                            # Merge on e.g. 'timestamp_str' + 'anomaly_score'
                            anomalies_df = anomalies_df.merge(
                                labeled_previous[["timestamp_str", "anomaly_score", "user_label"]],
                                on=["timestamp_str", "anomaly_score"],
                                how="left",
                                suffixes=("", "_old")
                            )
                            # Combine user_label if new is None but old is not
                            anomalies_df["user_label"] = anomalies_df["user_label"].combine_first(anomalies_df["user_label_old"])
                            anomalies_df.drop(columns=["user_label_old"], inplace=True, errors='ignore')

                    # Create columns in anomalies_df to identify numeric_id from file mapping
                    mapping_dict = load_file_mapping("data/file_mapping.json")
                    anomalies_df["original_file_id"] = anomalies_df["timestamp_str"].apply(parse_original_id)
                    anomalies_df["numeric_id"] = anomalies_df["original_file_id"].apply(lambda x: mapping_dict.get(x, None))
                    # Drop anomalies that don't match a numeric_id in the mapping
                    anomalies_df = anomalies_df.dropna(subset=["numeric_id"]).copy()
                    anomalies_df["numeric_id"] = anomalies_df["numeric_id"].astype(int)
                    
                    if anomalies_df.empty:
                        st.warning("All anomalies were filtered out because no matching numeric_id was found in file_mapping.json.")
                    else:
                        # Group by numeric_id
                        grouped = anomalies_df.groupby("numeric_id")
                        # Convert to a dict of dataframes
                        anomaly_groups = {nid: g.reset_index(drop=True) for nid, g in grouped}

                        # Initialize session state
                        if "labeled_groups" not in st.session_state:
                            st.session_state.labeled_groups = anomaly_groups
                        if "file_ids_in_order" not in st.session_state:
                            st.session_state.file_ids_in_order = sorted(anomaly_groups.keys())
                        if "file_idx" not in st.session_state:
                            st.session_state.file_idx = 0
                        if "anomaly_idx" not in st.session_state:
                            st.session_state.anomaly_idx = 0

                        # --------------------------------------------------------------------------------
                        # (Modification #2) 
                        # If labeling is already partially done, jump to the first unlabeled entry
                        # so that labeling continues from the next unlabeled row.
                        found_unlabeled = False
                        # If there's no unlabeled row at all in the entire dataset, we'll skip everything
                        for i, fid in enumerate(st.session_state.file_ids_in_order):
                            df_fid = st.session_state.labeled_groups[fid]
                            unlabeled_indices = df_fid[df_fid["user_label"].isna()].index
                            if len(unlabeled_indices) > 0:
                                st.session_state.file_idx = i
                                st.session_state.anomaly_idx = unlabeled_indices[0]
                                found_unlabeled = True
                                break
                        if not found_unlabeled:
                            # Means all labeled for all files
                            st.session_state.file_idx = len(st.session_state.file_ids_in_order)
                        # --------------------------------------------------------------------------------

                        def save_labeled_anomalies():
                            """
                            Gather all labeled groups from session state and save
                            to labeled_anomalies.csv.
                            """
                            all_dfs = []
                            for fid in st.session_state.labeled_groups:
                                all_dfs.append(st.session_state.labeled_groups[fid])
                            labeled_all = pd.concat(all_dfs, ignore_index=True)
                            labeled_all.to_csv("temp_mlflow/labeled_anomalies.csv", index=False)

                        # If we've labeled all files, show a success
                        if st.session_state.file_idx >= len(st.session_state.file_ids_in_order):
                            st.success("No more anomalies to label! All files have been processed.")
                            if st.button("Save labeled anomalies to CSV"):
                                save_labeled_anomalies()
                                st.success("Labeled anomalies saved to temp_mlflow/labeled_anomalies.csv.")
                            return

                        current_file_id = st.session_state.file_ids_in_order[st.session_state.file_idx]
                        current_df = st.session_state.labeled_groups[current_file_id]

                        # If we've gone past the last anomaly for this file, move to next file
                        if st.session_state.anomaly_idx >= len(current_df):
                            st.session_state.file_idx += 1
                            st.session_state.anomaly_idx = 0
                            st.rerun()

                        if len(current_df) == 0:
                            st.warning(f"No anomalies for file_id={current_file_id}. Moving on.")
                            st.session_state.file_idx += 1
                            st.rerun()

                        # Let user pick window sizes
                        st.subheader("Labeling Controls")
                        ccol1, ccol2 = st.columns([3, 1], gap="large")
                        with ccol2:
                            window_before = st.number_input("Rows before anomaly", min_value=0, value=20, step=1)
                            window_after = st.number_input("Rows after anomaly", min_value=0, value=20, step=1)
                        
                        anomaly_idx = st.session_state.anomaly_idx
                        anomaly_row = current_df.iloc[anomaly_idx]

                        # Prepare main layout
                        with ccol1:
                            # Display file info
                            possible_keys = [k for k, v in mapping_dict.items() if v == current_file_id]
                            if possible_keys:
                                file_key_str = possible_keys[0]
                            else:
                                file_key_str = f"ID={current_file_id}"
                            
                            st.markdown(
                                f"### File {file_key_str}.csv (numeric_id={current_file_id})"
                                f" - Anomaly {anomaly_idx+1} / {len(current_df)}"
                            )
                            score = anomaly_row.get("anomaly_score", None)
                            dist = anomaly_row.get("distance_from_threshold", None)
                            threshold = anomaly_row.get("threshold", None)
                            timestamp_str = anomaly_row["timestamp_str"]

                            st.write(f"**timestamp_str**: {timestamp_str}")
                            st.write(f"**Anomaly Score**: {score}")
                            st.write(f"**Threshold**: {threshold}")
                            st.write(f"**Distance**: {dist}")

                            # For clarity, let's show original vs predicted columns if needed
                            columns_to_hide = {
                                "timestamp_str", "anomaly_score", 
                                "distance_from_threshold", "threshold",
                                "original_file_id", "numeric_id", "user_label"
                            }
                            anomaly_display_cols = [c for c in anomaly_row.index if c not in columns_to_hide]
                            orig_cols = [c for c in anomaly_display_cols if not c.endswith("_p")]
                            pred_cols = [c for c in anomaly_display_cols if c.endswith("_p")]

                            if orig_cols or pred_cols:
                                st.write("**Feature Values**")
                                combined_df = pd.DataFrame({
                                    "Original": anomaly_row[orig_cols].values if orig_cols else [],
                                    "Predicted": anomaly_row[pred_cols].values if pred_cols else []
                                })
                                row_labels = orig_cols if orig_cols else pred_cols
                                if len(row_labels) == len(combined_df):
                                    combined_df.index = row_labels
                                st.dataframe(combined_df, use_container_width=True)

                        # --- Show a window of unscaled data around this anomaly ---
                        file_data = load_parquet_for_file(current_file_id, st.session_state.params.parquet_path)
                        if file_data is not None and not file_data.empty:
                            matched_rows = file_data[file_data["timestamp_str"] == timestamp_str]
                            if len(matched_rows) == 0:
                                st.warning(f"No match for timestamp_str={timestamp_str} in file_id={current_file_id}.")
                            else:
                                matched_idx = matched_rows.index[0]
                                start_idx = max(0, matched_idx - window_before)
                                end_idx = min(len(file_data), matched_idx + window_after + 1)

                                df_window = file_data.iloc[start_idx:end_idx].copy()
                                # remove columns we don't want to see
                                drop_cols = ["file_id", "timestamp_str"]
                                for dc in drop_cols:
                                    if dc in df_window.columns:
                                        df_window.drop(columns=[dc], inplace=True)
                                # unscale
                                df_window = unscale_features(df_window)

                                def highlight_anomaly_row(row):
                                    return [
                                        "background-color: yellow" if row.name == matched_idx else ""
                                        for _ in row
                                    ]

                                st.markdown(
                                    f"**Data Window** (± {window_before} / {window_after} rows around index {matched_idx})"
                                )
                                styled_window = df_window.style.apply(highlight_anomaly_row, axis=1)
                                st.dataframe(styled_window, use_container_width=True)
                        else:
                            st.warning("No data found in the Parquet for this file.")

                        # Labeling buttons
                        with ccol2:
                            st.write("---")
                            anomaly_button = st.button("Mark ANOMALY")
                            normal_button = st.button("Mark NORMAL")
                            # (Modification #7) Rename partial save button to "Save to MLflow"
                            partial_save_button = st.button("Save to MLflow")

                            if anomaly_button:
                                current_df.loc[anomaly_idx, "user_label"] = True
                                st.session_state.labeled_groups[current_file_id] = current_df
                                st.session_state.anomaly_idx += 1
                                # Immediately save partial
                                save_labeled_anomalies()
                                st.rerun()

                            if normal_button:
                                current_df.loc[anomaly_idx, "user_label"] = False
                                st.session_state.labeled_groups[current_file_id] = current_df
                                st.session_state.anomaly_idx += 1
                                # Immediately save partial
                                save_labeled_anomalies()
                                st.rerun()

                            if partial_save_button:
                                st.session_state.labeled_groups[current_file_id] = current_df
                                save_labeled_anomalies()
                                # Also log to MLflow
                                mlflow.log_artifact("temp_mlflow/labeled_anomalies.csv", artifact_path="labeled_results", run_id=run_id)
                                st.success("Labels saved to MLflow.")

                        st.session_state.labeled_groups[current_file_id] = current_df

    # ------------------- TAB 2: Attention Matrices ----------------------------
    with tabs[2]:
        st.header("Attention Matrices")
        st.write("Displays attention matrices from the loaded model. Also logs them to MLflow each time.")

        if "model" not in st.session_state:
            st.warning("Please load a model first in the 'MLflow Model Loader' tab.")
        else:
            max_plots = st.slider("Max Batches to Plot", 1, 10, 2)
            max_heads = st.slider("Max Heads per Batch", 1, 12, 2)
            sample_idx = st.number_input("Sample Index in Batch", 0, 100, 0)

            if st.button("Generate Attention Plots"):
                with st.spinner("Generating attention plots..."):
                    val_loader = st.session_state.val_loader if "val_loader" in st.session_state else None
                    if val_loader is None:
                        st.warning("No validation loader found. Please run detection or load data first.")
                    else:
                        attention_plots = plot_attention_matrices(
                            model=st.session_state.model, 
                            dataloader=val_loader, 
                            device="cpu",
                            max_plots=max_plots,
                            max_heads=max_heads,
                            sample_idx=sample_idx
                        )
                        if not attention_plots:
                            st.warning("No attention plots generated.")
                        else:
                            for title, fig in attention_plots:
                                st.write(title)
                                st.pyplot(fig)
                                # Save each fig
                                fig_name = f"{title.replace(' ', '_')}_q_{str(st.session_state.params.q)[-2:]}_p_{st.session_state.params.p}_h_{st.session_state.params.n_heads}.png"
                                fig_path = os.path.join(st.session_state.params.output_dir, fig_name)
                                fig.savefig(fig_path)
                                plt.close(fig)
                                # Log to MLflow in the same run
                                mlflow.log_artifact(fig_path, artifact_path="attention_plots", run_id=run_id)
                            st.success("Attention plots generated and saved to MLflow.")

    # ------------------- TAB 3: Highest Score Anomalies -----------------------
    with tabs[3]:
        st.header("Highest Score Anomalies")
        st.write("Displays a configurable top-N anomalies from your detected_anomalies.csv (by anomaly_score).")
        topN = st.number_input("How many top anomalies to show:", min_value=1, max_value=10000, value=10)
        
        anomalies_csv_path = "temp_mlflow/detected_anomalies.csv"
        if not os.path.exists(anomalies_csv_path):
            st.warning("No anomalies CSV found. Please run detection first.")
        else:
            anomalies_df = pd.read_csv(anomalies_csv_path)
            if anomalies_df.empty:
                st.warning("No rows in anomalies CSV.")
            else:
                # Sort by anomaly_score descending, then show top N
                anomalies_df = anomalies_df.sort_values(by="anomaly_score", ascending=False)
                top_df = anomalies_df.head(topN)
                feature_cols = [c for c in top_df.columns if not c.endswith("_p")]
                st.dataframe(top_df[feature_cols].reset_index(drop=True), use_container_width=True)


def main():
    app_main()
    # End run when the script finishes if we have an active run
    active_run = mlflow.active_run()
    if active_run:
        mlflow.end_run()


if __name__ == "__main__":
    main()
