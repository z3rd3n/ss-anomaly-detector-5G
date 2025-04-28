
# Semi-supervised Anomaly Detection in 5G Layer 2 Messages

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch Version](https://img.shields.io/badge/pytorch-1.10%2B-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-CC%20BY%203.0%20DE-lightgrey.svg)](http://creativecommons.org/licenses/by/3.0/de)

## Overview

This project implements a novel semi-supervised deep learning framework for detecting anomalies in 5G New Radio (NR) Layer 2 messages, specifically focusing on the Physical Downlink Shared Channel (PDSCH) and its associated Hybrid Automatic Repeat Request (HARQ) processes. The goal is to identify transmission irregularities that can impact network performance and reliability, which is crucial for applications ranging from enhanced mobile broadband (eMBB) to ultra-reliable low-latency communications (URLLC).

The core model utilizes a Bidirectional Gated Recurrent Unit (BiGRU) combined with a Multi-Head Attention mechanism and an innovative classification head that incorporates reconstruction error. This semi-supervised approach effectively addresses the challenges of extreme class imbalance common in telecommunication datasets (where anomalies are rare) and leverages unlabeled data patterns.

This repository contains the implementation of the main proposed model (`main.py`) as described in the Master's Thesis, along with a baseline LSTM Autoencoder model (`lstm.py`) for comparison.

## Key Features

*   **Semi-Supervised Learning:** Combines reconstruction-based learning (self-supervised) with supervised classification using limited anomaly labels.
*   **Advanced Architecture:** Employs BiGRU for sequential dependency modeling and Multi-Head Attention for focusing on relevant temporal patterns.
*   **Imbalance Handling:** Addresses extreme class imbalance (~99% normal) through:
    *   Anomaly-centered sequence creation during training.
    *   Weighted Binary Cross-Entropy loss.
    *   A custom regularization term promoting detection diversity across different anomaly types.
*   **Reconstruction-Enhanced Classification:** Uses the reconstruction error from the BiGRU pathway as an explicit input feature to the classification head, significantly improving F1-score.
*   **Specific Anomaly Detection:** Designed to identify four key Layer 2 HARQ anomalies:
    1.  **Unnecessary Retransmission:** Packet retransmitted despite successful prior reception.
    2.  **Missing Retransmission:** Packet fails CRC, but expected retransmission doesn't occur.
    3.  **New Data with No Retransmission:** Packet fails CRC, but new data is sent instead of retransmission.
    4.  **Maximum Retransmission Limit Violation:** Packet discarded after exceeding the maximum allowed retransmissions.
*   **Baseline Comparison:** Includes an LSTM Autoencoder baseline trained only on normal data for performance comparison.
*   **Detailed Evaluation:** Provides comprehensive evaluation metrics including F1-score, AUC, Accuracy, per-class detection rates, and confusion matrices.
*   **False Positive Analysis:** Includes functionality (in `main.py`) to log and analyze false positives to potentially identify unlabeled anomalies (requires further domain expertise).

## Architecture

The main model (`BiGRUAnomalyDetector` in `main.py`) follows a multi-task architecture:

1.  **Embedding Layer (`EmbeddingLayer`):**
    *   Embeds each categorical/ordinal input feature (SFN, Slot, HARQ, MCS, CRC, ReTx, NDI) into a dense vector space.
    *   Optionally adds learned positional encoding to retain sequence order information.
    *   Concatenates feature embeddings.

2.  **Sequential Encoder (BiGRU):**
    *   Processes the sequence of embedded features using a Bidirectional GRU to capture temporal dependencies from both past and future contexts (configurable to be unidirectional).

3.  **Reconstruction Path:**
    *   A linear layer projects the BiGRU hidden states back to the original embedding dimension.
    *   Calculates the Mean Squared Error (MSE) between the original embeddings and the reconstructed ones. This *reconstruction loss* is part of the total loss function.
    *   The *per-timestep MSE* is also used as a feature for the classifier.

4.  **Classification Path (`AttentionBinaryClassifier`):**
    *   **Multi-Head Self-Attention:** Applied to the BiGRU hidden states to allow the model to weigh the importance of different timesteps within the sequence.
    *   **Residual Connection & Normalization:** Standard Transformer-style residual connection and layer normalization are applied after attention.
    *   **Error Integration:** The per-timestep reconstruction error (from step 3) is concatenated with the attention output and processed through a linear layer.
    *   **Feed-Forward Layers:** Fully connected layers with GELU activation and Layer Normalization process the combined features.
    *   **Output Layer:** A final linear layer outputs a single logit per timestep, representing the likelihood of an anomaly.

5.  **Combined Loss Function (`AnomalyTypeLoss`):**
    *   `L_total = λ_rec * L_rec + λ_bin * L_bin + λ_reg * L_reg`
    *   `L_rec`: MSE reconstruction loss (encourages learning normal patterns).
    *   `L_bin`: Weighted Binary Cross-Entropy loss on the logits (handles normal/anomaly imbalance).
    *   `L_reg`: Custom regularization term encouraging balanced detection across the four known anomaly types, preventing focus only on the most frequent ones.

### Baseline Model (`LSTMAutoencoder` in `lstm_ae_baseline.py`)

*   Uses the same `EmbeddingLayer`.
*   Employs an LSTM Encoder-Decoder architecture.
*   Trained *only* on sequences containing normal data points.
*   Uses reconstruction error (MSE) for anomaly detection: points with error above a determined threshold are flagged as anomalies.

## Dataset

*   **Source:** Real-world 5G NR protocol data collected from an operational network using Rohde & Schwarz ROMES software. Data focuses on Layer 2 PDSCH messages.
*   **Format:** Processed and stored in Apache Parquet format (`.parquet`).
*   **Features:**
    *   `SFN` (System Frame Number): 0-1023 (Cyclic Integer)
    *   `Slot`: 0-159 (Cyclic Integer, depends on numerology)
    *   `HARQ ID`: 0-15 (Integer)
    *   `MCS` (Modulation and Coding Scheme): 0-31 (Integer)
    *   `CRC` (Cyclic Redundancy Check): 0 (Fail) or 1 (Pass) (Binary)
    *   `ReTx` (Retransmission Counter): 0-8 (Counter)
    *   `NDI` (New Data Indicator): 0 or 1 (Binary Toggle)
    *   `insight`: String describing detected anomalies (used for labeling).
    *   `timestamp_str`: Timestamp for each message.
*   **Sequence Creation (`AnomalySequenceDataset` in `dataset.py`):**
    *   **Training:** Uses an *anomaly-centered* approach. For each anomaly found, a sequence of length `seq_len` ending at the anomaly is created. Additionally, a fraction (e.g., 15%) of sequences containing *only* normal points are added to ensure the model learns normal behavior and address imbalance.
    *   **Evaluation/Testing:** Uses standard non-overlapping sliding windows of length `seq_len` across the entire dataset.
*   **Baseline Dataset (`NormalSequenceDataset` in `lstm_ae_baseline.py`):**
    *   **Training:** Creates overlapping sequences containing *only* normal data points.
    *   **Evaluation/Testing:** Same as the main model (non-overlapping windows across all data).

## Setup

1.  **Prerequisites:**
    *   Python 3.8 or higher
    *   PyTorch 1.10 or higher (with CUDA support recommended for GPU acceleration)
    *   Git

2.  **Clone Repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-directory>
    ```

3.  **Create Virtual Environment (Recommended): or use the Docker Container**
    ```bash
    python -m venv venv
    # Activate environment:
    # Windows: .\venv\Scripts\activate
    # Linux/macOS: source venv/bin/activate

    # inside .devcontainer there is devcontainer.json
    # open it using VsCode if you want to use docker container
    ```

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Ensure `requirements.txt` exists and lists necessary packages like `torch`, `numpy`, `pyarrow`, `tqdm`, `matplotlib`, `scikit-learn`)*

5.  **Dataset:**
    *   Place your training (`unscaled_pdsch_val.parquet`) and testing (`unscaled_pdsch_val_min.parquet`) Parquet files in the root directory or update the paths in the configuration dictionaries within the Python scripts.

## Usage

Both the main model and the baseline model are configured and run via their respective Python scripts.

### Main Model (BiGRU + Attention)

1.  **Configure:** Open `main.py` and adjust the `config` dictionary as needed:
    *   `train_parquet_path`, `test_parquet_path`: Paths to your data.
    *   `seq_len`, `batch_size`, `num_workers`: Dataset parameters.
    *   `embedding_dim`, `hidden_dim`, `num_layers`, etc.: Model hyperparameters.
    *   `learning_rate`, `num_epochs`, `early_stopping_patience`: Training parameters.
    *   `binary_weight`, `reconstruction_weight`, `regularization_weight`: Loss function weights.
    *   `threshold`: Probability threshold for classifying anomalies during evaluation.

2.  **Run Training and Evaluation:**
    ```bash
    python main.py
    ```

3.  **Outputs:**
    *   Console logs showing training progress, validation results per epoch (Losses, F1, AUC, Accuracy, Confusion Matrix, Per-Class Detection Rates).
    *   `anomaly_detector.log`: Detailed log file.
    *   `models/`: Directory containing saved model checkpoints (`.pth`) per epoch.
    *   `best_model.pth`: The model state dictionary corresponding to the best validation F1 score.
    *   `loss_curves.png`: Plot of training and validation loss over epochs.
    *   `false_positives_*.csv`: CSV files containing details of false positive predictions for analysis (generated if logging is enabled appropriately in the `train_model` function).


## Results Summary

The proposed semi-supervised BiGRU+Attention model demonstrates strong performance in detecting Layer 2 anomalies:

*   Achieves a high overall binary F1-score (e.g., ~82.78% in the thesis example) on the test set, indicating a good balance between precision and recall despite class imbalance.
*   Shows excellent detection rates for normal events (>99%) and high rates for Type 1 (Unnecessary Retx, e.g., ~97%) and Type 4 (Max Retx, e.g., ~100%) anomalies.
*   Significantly improves detection of rare anomaly types (Type 2 and Type 3) compared to standard approaches, thanks to the custom regularization term (e.g., ~60% for Type 2, ~82% for Type 3 in the thesis).
*   Incorporating reconstruction error as a feature boosts the F1 score substantially (by ~15% in ablation studies).
*   Outperforms the LSTM Autoencoder baseline significantly, especially in detecting rarer and more subtle anomaly types.

*For detailed results, ablation studies, and analysis, please refer to the associated Master's Thesis.*

## Limitations and Future Work

(Based on thesis Chapter 4)

**Limitations:**

*   **Single Channel Focus:** Current implementation analyzes only the PDSCH channel.
*   **Limited Features:** Uses only seven core Layer 2 parameters, potentially missing complex interactions visible with more data (e.g., CSI, power levels).
*   **Offline Learning:** Operates on historical data, not real-time streams.
*   **Threshold Dependency:** Requires careful tuning of the classification threshold.
*   **Interpretability:** While attention helps, deeper root cause analysis remains challenging.

**Future Work:**

*   **Multi-Channel Integration:** Extend the model to incorporate PDCP DL/UL, measurement reports, etc., for cross-layer anomaly detection.
*   **Online Learning:** Develop an adaptive version for real-time network monitoring.
*   **Enhanced Interpretability:** Integrate more advanced explanation techniques (e.g., SHAP, LIME) for root cause analysis.
*   **Active Learning:** Incorporate strategies to request expert labels for ambiguous cases, improving performance over time.
*   **Broader Validation:** Test across diverse network deployments and conditions.

## License

This work is licensed under the Creative Commons Attribution 3.0 Germany License. See [CC BY 3.0 DE](http://creativecommons.org/licenses/by/3.0/de).


