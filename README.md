# README

** Privacy-Preserving Federated Learning Framework for Collaborative Intrusion Detection in Distributed IIoT Environments**

## Overview

FedXAI-IDS is a research-driven framework for federated intrusion detection in distributed Industrial IoT (IIoT) environments. It trains a residual MLP across clients whose raw traffic data stays local, and studies how data heterogeneity (IID vs. class-wise Dirichlet non-IID partitions) and the optimiser (FedAvg vs. FedProx) affect detection performance. Gradient × Input attribution is used to analyse which features drive the model's predictions.

Local data storage avoids centralising traffic data, but it does not by itself provide a formal privacy guarantee against leakage from model updates.

## Key Features

- Federated Learning-based Intrusion Detection (residual MLP)
- Data Heterogeneity Evaluation:
  * IID partitioning
  * Class-wise Dirichlet non-IID partitioning (α = 0.1 | 0.5 | 1.0)
- Federated Optimisers: FedAvg and FedProx (μ = 0.01)
- Centralised Baselines: Logistic Regression and centralised MLP
- Explainability: Gradient × Input attribution (global, class-conditional, IID vs. non-IID)
- Client Diagnostics: attack-rate SD, update ℓ₂-norm, cosine similarity, learning curves
- Ablations: Dirichlet α, client count (5 | 10 | 20), training-data fraction
- Four Datasets: CIC-IoT2023, Edge-IIoTset, UNSW-NB15, Custom-Attack-Suite
- Multi-seed evaluation (10 production seeds, mean ± SD)

## Installation

### Clone Repository

```text
git clone https://github.com/<your-username>/FedXAI-IDS.git
cd FedXAI-IDS
```

### Install Dependencies

```text
pip install -r requirements.txt
```

Or install manually:

```text
pip install numpy pandas scipy matplotlib torch pyarrow openpyxl
```

A CUDA-capable GPU is recommended (automatic mixed precision is used when available); CPU execution is supported with `--device cpu`.

## Dataset

Dataset location:

```text
Dataset/
```

The raw datasets are **not** redistributed in this repository. The experiments use four datasets:

| Dataset | Source | Expected file(s) in `Dataset/` |
|---|---|---|
| Edge-IIoTset (worker 0) | Public benchmark | `ML-EdgeIIoT-dataset.csv` |
| CIC-IoT2023 (worker 1) | Public benchmark | `ciciot23.csv` |
| UNSW-NB15 (worker 2) | Public benchmark | `UNSW_NB15_training-set.parquet`, `UNSW_NB15_testing-set.parquet` |
| Custom-Attack-Suite (worker 3) | In-house | 13 CSV files (below) |

**Custom-Attack-Suite** combines 13 traffic files labelled at file level (benign = 0, attack = 1):

- Benign (10): `Distance.csv`, `Flame_Sensor.csv`, `Heart_Rate.csv`, `IR_Receiver.csv`, `Modbus.csv`, `phValue.csv`, `Soil_Moisture.csv`, `Sound_Sensor.csv`, `Temperature_and_Humidity.csv`, `Water_Level.csv`
- Attack (3): `DDoS_HTTP_Flood_attack.csv`, `Ransomware_attack.csv`, `SQL_injection_attack.csv`

After preprocessing: 32 features, 7,968,739 training and 1,992,184 test samples, 2.92 % attack rate (Table 3).

## Setup Instructions

### Step 1 — Download the Datasets

Obtain CIC-IoT2023, Edge-IIoTset and UNSW-NB15 from their original sources and place the files listed above in `Dataset/`.

### Step 2 — Verify Path

The script reads from `Dataset/` by default. To use another folder, pass `--data-dir`:

```text
python code/fedxai_ids_gpu_4workers.py --worker-id 0 --data-dir Dataset
```

### Step 3 — Run a Smoke Test

```text
python code/fedxai_ids_gpu_4workers.py --smoke --worker-id 0 --data-dir Dataset
```

Smoke mode uses a stratified subset and is for pipeline validation only. Do not mix smoke outputs with production results.

### Step 4 — Run the Full Experiments (one worker per dataset)

```text
python code/fedxai_ids_gpu_4workers.py --worker-id 0 --data-dir Dataset --device cuda
python code/fedxai_ids_gpu_4workers.py --worker-id 1 --data-dir Dataset --device cuda
python code/fedxai_ids_gpu_4workers.py --worker-id 2 --data-dir Dataset --device cuda
python code/fedxai_ids_gpu_4workers.py --worker-id 3 --data-dir Dataset --device cuda
```

Defaults match the manuscript: 10 seeds, 20 clients, 30 rounds, 5 local epochs, batch size 512, Adam (lr 1e-3), α = 0.5, FedProx μ = 0.01. Outputs are written to `supercomputer_outputs/`.

## Project Structure

```text
FedXAI-IDS/
│
├── code/
│   └── fedxai_ids_gpu_4workers.py   # GPU pipeline (one worker per dataset)
│
├── figures/                         # Results figures (Figs. 3–7)
├── tables/                          # Results tables (Tables 3–13) as CSV + all_tables.xlsx
├── README.md
├── requirements.txt
├── CITATION.cff
└── .zenodo.json
```

## Pipeline Workflow

1. Dataset-specific cleaning (duplicates, missing values, identifiers)
2. Deterministic categorical encoding
3. Stratified train/test split before federation; scaler fitted on the training set only
4. IID or class-wise Dirichlet (α) partition across K clients
5. Local training of the same residual MLP on each client (GPU, AMP when CUDA)
6. Aggregation:
   * FedAvg (sample-count weighted)
   * FedProx (proximal term, μ = 0.01)
7. Client diagnostics:
   * Client attack rates
   * Update ℓ₂-norm
   * Cosine similarity
8. Gradient × Input attribution:
   * Global
   * Dependence
   * Class-conditional
   * IID vs. non-IID
9. Multi-seed mean ± SD; all tables generated from one master results registry

## Model

| Layer | Units | Activation / regularisation |
|---|---|---|
| Input | d | Scaled feature vector |
| Linear | 256 | ReLU, LayerNorm, dropout 0.3 |
| Linear | 128 | ReLU with residual addition |
| Linear | 64 | ReLU, dropout 0.2 |
| Output | 1 | Sigmoid (binary classification) |

## Evaluation Metrics

- **Accuracy** — overall classification performance (can be misleading on imbalanced data)
- **Precision / Recall** — attack-prediction quality and attack coverage
- **F1 Score** — harmonic mean of precision and recall
- **FPR** — share of benign samples flagged as attacks
- **ROC-AUC** — threshold-independent ranking quality
- **Balanced Accuracy** — equal weight on both classes
- **IID vs. Non-IID Comparison** — effect of Dirichlet partitioning
- **Ablations** — α, client count and training-data fraction

## Experimental Results

Reported as mean ± standard deviation over seeds (Section 4 of the manuscript; full tables in `tables/`).

### Centralised Baselines (F1-score)

| Dataset | Logistic Regression | Centralised MLP |
|---|---|---|
| CIC-IoT2023 | 0.9907 | 0.9959 |
| Edge-IIoTset | 0.9331 | 0.9956 |
| UNSW-NB15 | 0.7947 | 0.8734 |
| Custom-Attack-Suite | 0.0000 | 0.8154 |

### Federated vs. Centralised (F1-score)

| Dataset | Centralised MLP | FedAvg-IID | FedAvg-NonIID | FedProx-NonIID |
|---|---|---|---|---|
| CIC-IoT2023 | 0.9959 | 0.9950 | 0.9914 | 0.9909 |
| Edge-IIoTset | 0.9956 | 0.9902 | 0.9908 | 0.9885 |
| UNSW-NB15 | 0.8734 | 0.8705 | 0.8545 | 0.8545 |
| Custom-Attack-Suite | 0.8154 | 0.8153 | 0.8520 | 0.8803 |

### Dirichlet α Sensitivity (F1-score)

| Dataset | α = 0.1 | α = 0.5 | α = 1.0 |
|---|---|---|---|
| CIC-IoT2023 | 0.9881 | 0.9902 | 0.9929 |
| Edge-IIoTset | 0.9472 | 0.9828 | 0.9893 |
| UNSW-NB15 | 0.8511 | 0.8538 | 0.8511 |
| Custom-Attack-Suite | 0.5968 | 0.8470 | 0.8134 |

### Client-Count Scaling (F1-score, α = 0.5)

| Dataset | 5 clients | 10 clients | 20 clients |
|---|---|---|---|
| CIC-IoT2023 | 0.9921 | 0.9912 | 0.9902 |
| Edge-IIoTset | 0.9811 | 0.9857 | 0.9828 |
| UNSW-NB15 | 0.8622 | 0.8526 | 0.8538 |
| Custom-Attack-Suite | 0.8152 | 0.8135 | 0.8470 |

These differences are descriptive. No paired statistical test is reported, so no statistical superiority is claimed.

## Key Insights

- Federated training approaches centralised F1 on CIC-IoT2023 and Edge-IIoTset
- The effect of non-IID partitioning is dataset- and metric-dependent (e.g. lower balanced accuracy on CIC-IoT2023 and UNSW-NB15)
- FedProx is not consistently better than FedAvg; it gives the highest F1 on Custom-Attack-Suite
- No consistent benefit from increasing the number of clients
- α = 0.5 is not best on every dataset
- Training curves improve mostly in the first 5–8 rounds, then plateau (observation, not a convergence proof)
- On Custom-Attack-Suite (2.92 % attacks), accuracy alone is misleading; use F1, recall and balanced accuracy

## Research Contributions

- Common residual-MLP evaluation of centralised, FedAvg and FedProx training across four IDS datasets
- Systematic IID vs. class-wise Dirichlet non-IID comparison with client diagnostics
- Ablations over Dirichlet α, client count and training-data fraction
- Gradient × Input attribution for federated IDS (global, class-conditional, IID vs. non-IID)
- Class-sensitive evaluation on an imbalanced in-house dataset (Custom-Attack-Suite)

## Limitations

- Controlled multi-worker setting; device heterogeneity, network latency and client dropout are not modelled
- Binary classification only
- No formal privacy guarantee (no differential privacy or secure aggregation)
- Attribution does not establish causality
- Some F1 values in the manuscript draft differ from the abstract and response-to-reviewers file; regenerate tables from the single verified results registry before the final release

## Important Notes

- Place datasets in `Dataset/` (or use `--data-dir`) before running
- Use relative paths; do not use absolute paths such as `C:\Users...` or `/home/user/...`
- Run each dataset as a separate worker (`--worker-id 0..3`)
- Keep smoke-test outputs separate from production results

## Common Errors

| Error | Cause | Fix |
|---|---|---|
| FileNotFoundError ("Expected under ...") | Dataset file missing from `Dataset/` | Download the file and use the exact name listed above |
| CUDA error / no GPU | No CUDA device available | Add `--device cpu` (slower) |
| Parquet read error (UNSW-NB15) | `pyarrow` not installed | `pip install pyarrow` |
| Wrong dataset processed | Wrong `--worker-id` | 0 = Edge-IIoTset, 1 = CIC-IoT2023, 2 = UNSW-NB15, 3 = Custom-Attack-Suite |

## Authors

| Name |
|---|
| Shailendra Mishra |
| Rayyanah Khalaf Alnoban |
| Fahad Alodhyani |

Department of Computer Engineering / Information Technology, College of Computer and Information Sciences, Majmaah University, Al Majmaah 11952, Saudi Arabia.

## DOI

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

Placeholder: replace `10.5281/zenodo.XXXXXXX` with the DOI that Zenodo assigns after the release is archived.

## License

This project is intended for academic and research use only.
