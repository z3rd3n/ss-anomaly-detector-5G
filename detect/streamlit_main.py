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

# --- Ensure the session_state variables exist before using them ---
if "detect_path" not in st.session_state:
    st.session_state.detect_path = ""

if "labeled_path" not in st.session_state:
    st.session_state.labeled_path = ""

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from detect.mlflow_utils import load_config_from_mlflow, download_artifact
from utils import load_checkpoint, plot_attention_matrices, unscale_features
from approaches.subAdjacent.trainer import detect_anomalies
from data.dataLoader import ParquetSequenceDataset, custom_collate_fn


@st.cache_data
def load_file_mapping(mapping_path: str = "data/file_mapping.json"):
    with open(mapping_path, "r") as f:
        return json.load(f)

@st.cache_data
def load_parquet_for_file(numeric_id_val: int, parquet_path: str = "data/scaled_pdsch.parquet"):
    if not os.path.exists(parquet_path):
        return pd.DataFrame()

    df = pd.read_parquet(
        parquet_path,
        filters=[("file_id", "=", numeric_id_val)]
    )
    return df

def parse_original_id(timestamp_str: str) -> str:
    file_part = timestamp_str.split(";")[0]  # e.g. "155.csv"
    return "".join(filter(str.isdigit, file_part))  # "155"

def app_main():
    # Increase logging level
    logging.basicConfig(level=logging.INFO)
    st.set_page_config(layout="wide", page_title="Anomaly Labeling + MLflow Demo")

    tabs = st.tabs([
        "MLflow Model Loader", 
        "Anomaly Detection & Labeling", 
        "Attention Matrices", 
        "Highest Score Anomalies"
    ])

    # -------------------------------------------------------------------------
    # TAB 0: MLflow Model Loader
    # -------------------------------------------------------------------------
    with tabs[0]:
        st.header("MLflow Model Loader")
        st.write("This section downloads the model from MLflow and sets up the environment.")
        tracking_uri = st.text_input(
            "MLflow Tracking URI (e.g. http://localhost:8000)",
            value="http://localhost:8000",
            help="Must be http(s) if using mlflow-artifacts store"
        )
        run_id = st.text_input("Enter MLflow run_id:", value="809606034f6e4682b2f327d249eff203", help="Paste the run_id from MLflow")
        
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

                    # 2) Download checkpoint
                    ckpt_path = download_artifact(run_id, "checkpoints/checkpoint_best.pt", st.session_state.temp_dir)
                    mlflow.set_tracking_uri(tracking_uri)
                    mlflow.set_experiment("subAdjacent")
                    
                    if mlflow.active_run() is None:
                        mlflow.start_run(run_id=run_id)

                    # 3) Build model from the config
                    st.session_state.model = st.session_state.params.build_model()

                    # Build optimizer
                    if (
                        hasattr(st.session_state.params, "optimizer") 
                        and st.session_state.params.optimizer == "AdamW"
                    ):
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
                    
                    # 4) Load checkpoint
                    load_checkpoint(st.session_state.model, st.session_state.optimizer, ckpt_path)
                    
                    st.success("Model & config loaded from MLflow. Params updated.")
                    st.write("Current parameters:")
                    st.json({key: value for key, value in vars(st.session_state.params).items()})

    # -------------------------------------------------------------------------
    # TAB 1: Anomaly Detection & Labeling
    # -------------------------------------------------------------------------
    with tabs[1]:
        st.header("Anomaly Detection & Labeling")
        st.write("This section runs anomaly detection, saves results, and allows labeling anomalies.")

        if "model" not in st.session_state:
            st.warning("Please go to the 'MLflow Model Loader' tab to load a model first.")
        else:
            st.subheader("Run Anomaly Detection with custom p and q")
            p_val = st.number_input("Percentile (p)", min_value=0, max_value=100, value=95, step=1)
            q_val = st.number_input("Quartile (q)", min_value=0.0, max_value=1.0, value=0.99, step=0.01)
            validation_ratio = st.number_input("Val Ratio", min_value=0.2, max_value=1.0, value=0.2, step=0.1)
            st.session_state.params.q = q_val
            st.session_state.params.p = p_val
            st.session_state.params.validation_ratio = validation_ratio
            
            # Construct detect/labeled paths
            st.session_state.detect_path = os.path.join(
                st.session_state.params.output_dir, 
                f"anomalies_p{st.session_state.params.p}q{str(st.session_state.params.q)[2:]}v{int(st.session_state.params.validation_ratio * 100)}.csv"
            )
            st.session_state.labeled_path = os.path.join(
                st.session_state.params.output_dir, 
                f"labeled_p{st.session_state.params.p}q{str(st.session_state.params.q)[2:]}v{int(st.session_state.params.validation_ratio * 100)}.csv"
            )

            # Create validation dataset & loader
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

            # Run detection button
            if st.button("Run Anomaly Detection"):
                with st.spinner("Running detection..."):
                    # Overwrite old CSVs
                    if os.path.exists(st.session_state.detect_path):
                        st.warning(f"Anomalies CSV {st.session_state.detect_path} exists. Overwriting.")
                        os.remove(st.session_state.detect_path)

                    if os.path.exists(st.session_state.labeled_path):
                        st.warning(f"Labeled CSV {st.session_state.labeled_path} exists. Overwriting.")
                        os.remove(st.session_state.labeled_path) 

                    anomalies_df, fig_path = detect_anomalies(
                        params=st.session_state.params, 
                        model=st.session_state.model, 
                        val_loader=val_loader
                    )

                    anomalies_df.to_csv(st.session_state.detect_path, index=False)
                    st.success(f"Anomalies saved to {st.session_state.detect_path}")
                    mlflow.log_artifact(st.session_state.detect_path, artifact_path="detection_results", run_id=run_id)
                    if fig_path and os.path.exists(fig_path):
                        mlflow.log_artifact(fig_path, artifact_path="plots", run_id=run_id)
                    st.session_state.detected_anomalies_df = anomalies_df

            st.write("---")
            # -------------- Start of Labeling UI --------------------------------
            if not os.path.exists(st.session_state.detect_path):
                st.warning("No anomalies CSV found. Please run detection first.")
            else:
                anomalies_df = pd.read_csv(st.session_state.detect_path)
                if "anomalous_features" not in anomalies_df.columns:
                    anomalies_df["anomalous_features"] = None
                if anomalies_df.empty:
                    st.warning("No rows in anomalies CSV.")
                else:
                    st.write(f"Loaded {len(anomalies_df)} anomalies from {st.session_state.detect_path} for labeling.")
                    
                    # Merge with labeled anomalies if it exists
                    if os.path.exists(st.session_state.labeled_path):
                        labeled_previous = pd.read_csv(st.session_state.labeled_path)
                        if "anomalous_features" in labeled_previous.columns:
                            anomalies_df = anomalies_df.merge(
                                labeled_previous[["timestamp_str", "anomaly_score", "anomalous_features"]],
                                on=["timestamp_str", "anomaly_score"],
                                how="left",
                                suffixes=("", "_old")
                            )
                            anomalies_df["anomalous_features"] = anomalies_df["anomalous_features"].combine_first(
                                anomalies_df["anomalous_features_old"]
                            )
                            anomalies_df.drop(columns=["anomalous_features_old"], inplace=True, errors='ignore')

                    # file_mapping-based grouping
                    mapping_dict = load_file_mapping("data/file_mapping.json")
                    anomalies_df["original_file_id"] = anomalies_df["timestamp_str"].apply(parse_original_id)
                    anomalies_df["numeric_id"] = anomalies_df["original_file_id"].apply(lambda x: mapping_dict.get(x, None))
                    anomalies_df = anomalies_df.dropna(subset=["numeric_id"]).copy()
                    anomalies_df["numeric_id"] = anomalies_df["numeric_id"].astype(int)
                    
                    if anomalies_df.empty:
                        st.warning("No matching numeric_id was found in file_mapping.json for these anomalies.")
                    else:
                        # Group by numeric_id
                        grouped = anomalies_df.groupby("numeric_id")
                        anomaly_groups = {nid: g.reset_index(drop=True) for nid, g in grouped}

                        # Store in session_state
                        st.session_state.labeled_groups = anomaly_groups
                        st.session_state.file_ids_in_order = sorted(anomaly_groups.keys())

                        # Indices
                        if "file_idx" not in st.session_state:
                            st.session_state.file_idx = 0
                        if "anomaly_idx" not in st.session_state:
                            st.session_state.anomaly_idx = 0

                        # Persist selection
                        if "selected_anomalous_features" not in st.session_state:
                            st.session_state.selected_anomalous_features = []
                        if "last_shown_anomaly" not in st.session_state:
                            st.session_state.last_shown_anomaly = (None, None)

                        def save_labeled_anomalies():
                            """
                            Gather labeled data from session_state and save them to CSV.
                            """
                            all_dfs = []
                            for fid in st.session_state.labeled_groups:
                                all_dfs.append(st.session_state.labeled_groups[fid])
                            labeled_all = pd.concat(all_dfs, ignore_index=True)

                            # Adjust final columns so we include all relevant features:
                            final_cols = [
                                "timestamp_str",
                                "SFN", "Slot", "CC", "HARQ", "MCS", "CRC", "ReTx", "NDI",
                                "threshold", "anomaly_score", "anomalous_features",
                            ]
                            # Ensure we have them; fill missing columns with ""
                            for c in final_cols:
                                if c not in labeled_all.columns:
                                    labeled_all[c] = ""
                            labeled_all = labeled_all[final_cols]

                            labeled_all.to_csv(st.session_state.labeled_path, index=False)
                            return labeled_all

                        def move_to_next_anomaly():
                            st.session_state.anomaly_idx += 1
                            st.session_state.selected_anomalous_features = []
                            st.session_state.last_shown_anomaly = (
                                st.session_state.file_idx,
                                st.session_state.anomaly_idx
                            )

                        # ---------------- Check if we have leftover files ----------
                        if st.session_state.file_idx >= len(st.session_state.file_ids_in_order):
                            st.success("No more anomalies to label! All files have been processed.")
                            return

                        current_file_id = st.session_state.file_ids_in_order[st.session_state.file_idx]
                        current_df = st.session_state.labeled_groups[current_file_id]

                        # If we've exceeded the anomalies in the current file, move to next file
                        if st.session_state.anomaly_idx >= len(current_df):
                            st.session_state.file_idx += 1
                            st.session_state.anomaly_idx = 0
                            st.rerun()

                        if len(current_df) == 0:
                            st.warning(f"No anomalies for file_id={current_file_id}. Moving on.")
                            st.session_state.file_idx += 1
                            st.session_state.anomaly_idx = 0
                            st.rerun()

                        anomaly_idx = st.session_state.anomaly_idx
                        anomaly_row = current_df.iloc[anomaly_idx]
                        timestamp_str = anomaly_row["timestamp_str"]

                        # Possibly load previously labeled features
                        already_labeled = anomaly_row.get("anomalous_features", "")
                        if (st.session_state.file_idx, st.session_state.anomaly_idx) != st.session_state.last_shown_anomaly:
                            if pd.notna(already_labeled) and len(already_labeled.strip()) > 0:
                                st.session_state.selected_anomalous_features = [
                                    f.strip() for f in already_labeled.split(",") if f.strip()
                                ]
                            else:
                                st.session_state.selected_anomalous_features = []
                            st.session_state.last_shown_anomaly = (
                                st.session_state.file_idx,
                                st.session_state.anomaly_idx
                            )

                        # --- Layout: main column vs. side column ---
                        ccol_main, ccol_side = st.columns([3, 1], gap="large")

                        with ccol_main:
                            # Show Feature Values in row format
                            st.markdown(f"### Anomaly {anomaly_idx+1} / {len(current_df)} for File ID={current_file_id}")
                            score = anomaly_row.get("anomaly_score", None)
                            dist = anomaly_row.get("distance_from_threshold", None)
                            threshold = anomaly_row.get("threshold", None)

                            st.write(f"**timestamp_str**: {timestamp_str}")
                            st.write(f"**Anomaly Score**: {score}")
                            st.write(f"**Threshold**: {threshold}")
                            st.write(f"**Distance**: {dist}")

                            # Original vs predicted columns
                            columns_to_hide = {
                                "timestamp_str", "anomaly_score", 
                                "distance_from_threshold", "threshold",
                                "original_file_id", "numeric_id", "anomalous_features"
                            }
                            anomaly_display_cols = [c for c in anomaly_row.index if c not in columns_to_hide]
                            orig_cols = [c for c in anomaly_display_cols if not c.endswith("_p")]
                            pred_cols = [c for c in anomaly_display_cols if c.endswith("_p")]

                            if orig_cols or pred_cols:
                                st.markdown("**Feature Values** (rows = Original / Predicted, columns = features)")
                                # Create a 2-row DataFrame: row1=Original values, row2=Predicted values
                                # We'll use `orig_cols` as the reference columns
                                # (If you need to unify or reorder columns, adjust accordingly)
                                df_feat = pd.DataFrame(
                                    [
                                        anomaly_row[orig_cols].values if orig_cols else [],
                                        anomaly_row[pred_cols].values if pred_cols else []
                                    ],
                                    index=["Original", "Predicted"],
                                    columns=orig_cols
                                )
                                st.dataframe(df_feat, use_container_width=True)

                            # Show unscaled data window
                            file_data = load_parquet_for_file(current_file_id, st.session_state.params.parquet_path)
                            if file_data is not None and not file_data.empty:
                                matched_rows = file_data[file_data["timestamp_str"] == timestamp_str]
                                if len(matched_rows) == 0:
                                    st.warning(f"No match for timestamp_str={timestamp_str} in file_id={current_file_id}.")
                                else:
                                    matched_idx = matched_rows.index[0]
                                    # We'll set defaults for window_before & window_after for demonstration
                                    # (though we do them in ccol_side)
                                    window_before = 20
                                    window_after = 20
                                    
                                    if "window_before" in st.session_state:
                                        window_before = st.session_state.window_before
                                    if "window_after" in st.session_state:
                                        window_after = st.session_state.window_after

                                    start_idx = max(0, matched_idx - window_before)
                                    end_idx = min(len(file_data), matched_idx + window_after + 1)

                                    df_window = file_data.iloc[start_idx:end_idx].copy()
                                    drop_cols = ["file_id", "timestamp_str"]
                                    for dc in drop_cols:
                                        if dc in df_window.columns:
                                            df_window.drop(columns=[dc], inplace=True)
                                    df_window = unscale_features(df_window)

                                    def highlight_anomaly_row(row):
                                        return [
                                            "background-color: yellow" if row.name == matched_idx else ""
                                            for _ in row
                                        ]

                                    st.markdown(f"**Data Window** (± {window_before} / {window_after} rows)")
                                    styled_window = df_window.style.apply(highlight_anomaly_row, axis=1)
                                    st.dataframe(styled_window, use_container_width=True)
                            else:
                                st.warning("No data found in the Parquet for this file.")

                        with ccol_side:
                            # Let user pick how many rows before/after
                            # Store them in session_state so the main col can use them
                            st.session_state.window_before = st.number_input(
                                "Rows before anomaly",
                                min_value=0, value=20, step=1
                            )
                            st.session_state.window_after = st.number_input(
                                "Rows after anomaly",
                                min_value=0, value=20, step=1
                            )

                            # Multiselect for anomalous features
                            st.markdown("**Anomalous Features**")
                            selected_features = st.multiselect(
                                label="",
                                options=["SFN", "Slot", "CC", "HARQ", "MCS", "CRC", "ReTx", "NDI"],
                                default=st.session_state.selected_anomalous_features,
                                key="feature_multiselect"
                            )

                            # Next anomaly button
                            if st.button("Next Anomaly"):
                                # Save selection
                                if selected_features:
                                    current_df.loc[anomaly_idx, "anomalous_features"] = ", ".join(selected_features)
                                else:
                                    current_df.loc[anomaly_idx, "anomalous_features"] = "Normal"
                                st.session_state.labeled_groups[current_file_id] = current_df

                                # Immediately save to CSV
                                labeled_all = save_labeled_anomalies()
                                mlflow.log_artifact(
                                    st.session_state.labeled_path,
                                    artifact_path="labeled_results",
                                    run_id=run_id
                                )

                                # Move on
                                move_to_next_anomaly()
                                st.rerun()

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
                                fig_name = (
                                    f"{title.replace(' ', '_')}"
                                    f"_q_{str(st.session_state.params.q)[-2:]}"
                                    f"_p_{st.session_state.params.p}"
                                    f"_v_{int(st.session_state.params.validation_ratio * 100)}"
                                    f"_h_{st.session_state.params.n_heads}.png"
                                )
                                fig_path = os.path.join(st.session_state.params.output_dir, fig_name)
                                fig.savefig(fig_path)
                                plt.close(fig)
                                # Log to MLflow in the same run
                                mlflow.log_artifact(fig_path, artifact_path="attention_plots", run_id=run_id)
                            st.success("Attention plots generated and saved to MLflow.")

    # ------------------- TAB 3: Highest Score Anomalies -----------------------
    with tabs[3]:
        st.header("Highest Score Anomalies")
        st.write(f"Displays a configurable top-N anomalies from your {st.session_state.detect_path}(by anomaly_score).")
        topN = st.number_input("How many top anomalies to show:", min_value=1, max_value=10000, value=10)
        
        if not os.path.exists(st.session_state.detect_path):
            st.warning("No anomalies CSV found. Please run detection first.")
        else:
            anomalies_df = pd.read_csv(st.session_state.detect_path)
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
