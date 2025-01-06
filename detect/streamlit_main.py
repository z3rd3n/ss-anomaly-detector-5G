import os
import logging
import streamlit as st
import pandas as pd
import torch
import mlflow
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
from detect.mlflow_utils import load_config_from_mlflow, download_artifact
from utils import load_checkpoint, plot_attention_matrices
from approaches.subAdjacent.trainer import detect_anomalies
from data.dataLoader import ParquetSequenceDataset

def app_main():
    # Increase logging level
    logging.basicConfig(level=logging.INFO)
    st.set_page_config(layout="wide", page_title="Anomaly Labeling + MLflow Demo")

    # For demonstration, let's have multiple pages in Streamlit using st.tabs
    tabs = st.tabs(["MLflow Model Loader", "Anomaly Detection & Labeling", "Attention Matrices", "Highest Score Anomalies"])
    
    with tabs[0]:
        st.header("MLflow Model Loader")
        st.write("This section downloads the model from MLflow and sets up the environment.")
        
        run_id = st.text_input("Enter MLflow run_id:", value="", help="Paste the run_id from MLflow")
        if st.button("Load Model & Config from MLflow"):
            if not run_id:
                st.error("Please provide a run_id.")
            else:
                with st.spinner("Downloading and loading model..."):
                    st.session_state.temp_dir = "temp_mlflow"
                    os.makedirs(st.session_state.temp_dir, exist_ok=True)

                    # 1) Download and import the config_class
                    ConfigClass = load_config_from_mlflow(run_id, st.session_state.temp_dir)
                    st.session_state.params = ConfigClass()
                    
                    # 3) Download checkpoint
                    st.session_state.params.output_dir = st.session_state.temp_dir
                    ckpt_path = download_artifact(run_id, "checkpoints/checkpoint_best.pt", st.session_state.temp_dir)
                    
                    # 4) Build model from the config
                    # We assume st.session_state.params has a build_model() method
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
                    
                    # 6) Connect to MLflow
                    mlflow.set_experiment("subAdjacent")
                    mlflow.start_run(run_id=run_id)
                    
                    st.success("Model & config loaded from MLflow. Params updated.")
                    st.write("Current parameters:")
                    st.json({key: value for key, value in vars(st.session_state.params).items()})

    with tabs[1]:
        st.header("Anomaly Detection & Labeling")
        st.write("This section runs anomaly detection (if not already done), loads the anomalies, and allows labeling.")
        
        if "model" not in st.session_state:
            st.warning("Please go to the 'MLflow Model Loader' tab to load a model first.")
        else:
            # Optionally let user run detection
            if st.button("Run Anomaly Detection"):
                with st.spinner("Running detection..."):
                    logging.info("Creating train and validation datasets...")
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
                        collate_fn=st.session_state.custom_collate_fn,
                        pin_memory=st.session_state.params.pin_memory,
                        drop_last=True
                    )

                    st.session_state.val_loader = val_loader
                    
                    anomalies_df = detect_anomalies(st.session_state.params, st.session_state.model, val_loader)
                    
                    # Save to CSV
                    output_csv = "detect/detected_anomalies.csv"
                    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
                    anomalies_df.to_csv(output_csv, index=False)
                    st.success(f"Anomalies saved to {output_csv}")
                    
                    # Log to MLflow the final q, p, plus a metric of how many anomalies found
                    mlflow.log_param("final_q", st.session_state.params.q)
                    mlflow.log_param("final_p", st.session_state.params.p)
                    mlflow.log_metric("num_anomalies", len(anomalies_df))
                    
                    # Possibly also log the CSV as an artifact
                    mlflow.log_artifact(output_csv, artifact_path="detection_results")
                    
                    # Done
                    st.session_state.detected_anomalies_df = anomalies_df

            st.write("---")
            # Next, let's do the same labeling code, 
            # but let's incorporate partial labeling from a previous CSV if it exists.

            anomalies_csv_path = "detect/detected_anomalies.csv"
            if not os.path.exists(anomalies_csv_path):
                st.warning("No anomalies CSV found. Please run anomaly detection first.")
            else:
                anomalies_df = pd.read_csv(anomalies_csv_path)
                if anomalies_df.empty:
                    st.warning("No rows in anomalies CSV.")
                else:
                    st.write(f"Loaded {len(anomalies_df)} anomalies from {anomalies_csv_path}")

                    # Check if there's a labeled_anomalies.csv from before
                    labeled_csv_path = "labeled_anomalies.csv"
                    if os.path.exists(labeled_csv_path):
                        labeled_previous = pd.read_csv(labeled_csv_path)
                        st.write(f"Found existing labeled CSV with shape: {labeled_previous.shape}")
                        
                        # Merge them on e.g. 'timestamp_str' (and possibly other columns).
                        # We'll only keep the user_label from the old if it matches.
                        if "user_label" in labeled_previous.columns:
                            # Let's do a left join
                            anomalies_df = anomalies_df.merge(
                                labeled_previous[["timestamp_str", "user_label"]],
                                on="timestamp_str", 
                                how="left", 
                                suffixes=("", "_old")
                            )
                            # If user_label already existed from old, keep it
                            anomalies_df["user_label"] = anomalies_df["user_label"].combine_first(anomalies_df["user_label_old"])
                            anomalies_df.drop(columns=["user_label_old"], inplace=True, errors='ignore')
                    
                    # Now we proceed with the original labeling logic
                    # Group by something if needed...
                    # But let's just let the user label row by row.

                    st.session_state.anomalies_df_for_label = anomalies_df.copy()
                    
                    # Let user pick how many items to label at once
                    num_to_label = st.number_input("Number of anomalies to show for labeling at once:", 
                                                   min_value=1, max_value=20, value=5)
                    # Filter for unlabeled
                    unlabeled_mask = st.session_state.anomalies_df_for_label["user_label"].isna()
                    df_unlabeled = st.session_state.anomalies_df_for_label[unlabeled_mask].head(num_to_label)
                    
                    st.write(f"Showing {len(df_unlabeled)} anomalies to label. (Out of {unlabeled_mask.sum()} unlabeled total.)")

                    if len(df_unlabeled) > 0:
                        for idx, row in df_unlabeled.iterrows():
                            with st.expander(f"Anomaly at row index {idx}, score={row['anomaly_score']:.2f}", expanded=False):
                                st.write(row)
                                # Buttons
                                c1, c2 = st.columns(2)
                                if c1.button(f"Mark ANOMALY {idx}"):
                                    st.session_state.anomalies_df_for_label.loc[idx, "user_label"] = True
                                if c2.button(f"Mark NORMAL {idx}"):
                                    st.session_state.anomalies_df_for_label.loc[idx, "user_label"] = False

                        if st.button("Save partial labels"):
                            st.session_state.anomalies_df_for_label.to_csv("labeled_anomalies.csv", index=False)
                            st.success("Partial labels saved to labeled_anomalies.csv")
                    else:
                        st.success("No unlabeled anomalies left to show. All done!")
                        st.write("If you want to re-label or see them all, check labeled_anomalies.csv.")
                        
    with tabs[2]:
        st.header("Attention Matrices")
        st.write("Displays attention matrices from the loaded model.")
        if "model" not in st.session_state:
            st.warning("Please load a model first in the 'MLflow Model Loader' tab.")
        else:
            max_plots = st.slider("Max Batches to Plot", 1, 10, 2)
            max_heads = st.slider("Max Heads per Batch", 1, 12, 2)
            sample_idx = st.number_input("Sample Index in Batch", 0, 100, 0)
            if st.button("Generate Attention Plots"):
                with st.spinner("Generating attention plots..."):
                    val_loader = st.session_state.val_loader

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
                            plt.close(fig)

    with tabs[3]:
        st.header("Highest Score Anomalies")
        st.write("Displays a configurable top-N anomalies from your detected_anomalies.csv (by anomaly_score).")
        topN = st.number_input("How many top anomalies to show:", min_value=1, max_value=10000, value=10)
        
        anomalies_csv_path = "detect/detected_anomalies.csv"
        if not os.path.exists(anomalies_csv_path):
            st.warning("No anomalies CSV found. Please run detection first.")
        else:
            anomalies_df = pd.read_csv(anomalies_csv_path)
            if anomalies_df.empty:
                st.warning("No rows in anomalies CSV.")
            else:
                # Show top N
                top_df = anomalies_df.head(topN)
                # Show only feature columns (we'll guess them by checking columns that do not end in _p, ignoring anomaly_score etc.)
                feature_cols = [c for c in top_df.columns if not c.endswith("_p")]
                st.dataframe(top_df[feature_cols].reset_index(drop=True), use_container_width=True)


def main():
    # In a real multi-page Streamlit app, you'd do st.sidebar or something else.
    # For a single-file approach, we'll just run the single function that sets up tabs.
    app_main()
    # End run when the script finishes if we have an active run
    active_run = mlflow.active_run()
    if active_run:
        mlflow.end_run()


if __name__ == "__main__":
    main()