# README

**FedXAI-IDS: Privacy-Preserving Federated Learning Framework for Collaborative Intrusion Detection in Distributed IIoT Environments**

Code, tables and figures accompanying the manuscript "Privacy-Preserving Federated Learning Framework for Collaborative Intrusion Detection in Distributed IIoT Environments".

**Authors:** Shailendra Mishra, Rayyanah Khalaf Alnoban, Fahad Alodhyani
(Majmaah University, Al Majmaah 11952, Saudi Arabia)

## DOI

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

Placeholder: replace `10.5281/zenodo.XXXXXXX` with the DOI that Zenodo assigns after the release is archived.

## Overview

FedXAI-IDS is a federated intrusion-detection framework built on a residual MLP (256 → 128 with residual shortcut → 64 → 1, LayerNorm, dropout 0.3/0.2). It studies collaborative learning under IID and class-wise Dirichlet non-IID client partitions, compares centralised training with FedAvg and FedProx, and uses Gradient × Input attribution to analyse which features drive predictions. Local storage of training data avoids data aggregation but provides no formal privacy guarantee on its own.

**Setup:** four datasets (CIC-IoT2023, Edge-IIoTset, UNSW-NB15, Custom-Attack-Suite); 20 clients (primary), 30 rounds, 5 local epochs, batch size 512, Adam (lr 1e-3), FedProx μ = 0.01, Dirichlet α = 0.5 (ablations: α ∈ {0.1, 0.5, 1.0}; 5/10/20 clients; training fractions 0.10–1.00); 10 production seeds. Configurations: Logistic Regression and centralised MLP baselines, FedAvg-IID, FedAvg-NonIID, FedProx-NonIID. Metrics: Accuracy, Precision, Recall, F1, FPR, ROC-AUC, Balanced Accuracy.

**Custom-Attack-Suite** is an in-house collection of 13 traffic files: 10 benign device files (Distance, Flame Sensor, Heart Rate, IR Receiver, Modbus, pH Value, Soil Moisture, Sound Sensor, Temperature & Humidity, Water Level) and 3 attack files (DDoS HTTP Flood, Ransomware, SQL Injection), labelled at file level for binary detection (32 features after preprocessing).

## Repository structure

```
FedXAI-IDS/
├── README.md
├── requirements.txt
├── CITATION.cff
├── .zenodo.json
├── code/
│   └── fedxai_ids_gpu_4workers.py   # GPU pipeline: one worker per dataset (0..3)
├── figures/                         # manuscript Results figures (Section 4)
└── tables/                          # manuscript Results tables (Section 4) as CSV + all_tables.xlsx
```

## Figures

| Manuscript | File | Content |
|---|---|---|
| Fig. 3 | `figures/fig3_cross_dataset_federated_performance.png` | Cross-dataset federated performance (F1-score, ROC-AUC, balanced accuracy; mean ± SD) |
| Fig. 4 | `figures/fig4_learning_curves_f1.png` | Learning curves, F1-score vs communication rounds (FedAvg-IID, FedAvg-NonIID, FedProx-NonIID) |
| Fig. 5 | `figures/fig5_noniid_update_diagnostics.png` | Non-IID update diagnostics (update ℓ₂-norm and cosine similarity) |
| Fig. 6 | `figures/fig6_dirichlet_alpha_sensitivity.png` | Dirichlet concentration-parameter sensitivity (F1-score vs α) |
| Fig. 7 | `figures/fig7_training_fraction_saturation.png` | F1-score and ROC-AUC vs training-data fraction |

Figs. 1–2 (framework and implementation workflow) belong to Section 3 and are not included here.

## Tables

| Manuscript | File | Caption |
|---|---|---|
| 3 | `tables/table03_dataset_characteristics.csv` | Dataset characteristics after preprocessing |
| 4 | `tables/table04_residual_mlp_architecture.csv` | Residual MLP architecture used in the experiments |
| 5 | `tables/table05_centralised_performance.csv` | Centralised classification performance |
| 6 | `tables/table06_federated_performance.csv` | Federated performance across the four datasets (mean ± standard deviation) |
| 7 | `tables/table07_iid_vs_noniid.csv` | Descriptive comparison of IID and non-IID performance |
| 8 | `tables/table08_dirichlet_alpha_f1.csv` | Dirichlet concentration-parameter sensitivity (F1-score) |
| 9 | `tables/table09_client_count_f1.csv` | Effect of client count on F1-score (α = 0.5) |
| 10 | `tables/table10_training_fraction.csv` | Performance at different training-data fractions |
| 11 | `tables/table11_centralised_vs_federated_f1.csv` | Centralised and federated F1-score comparison |
| 12 | `tables/table12_research_question_alignment.csv` | Alignment between research questions and experimental findings |
| 13 | `tables/table13_sota_comparison.csv` | State-of-the-art comparison with closely related studies |

All eleven tables are also in `tables/all_tables.xlsx` (one sheet per table). Tables 1–2 belong to Sections 2–3 and are not included.

## Reproducing the experiments

```bash
pip install -r requirements.txt

# 1. Place the datasets under ./Dataset (see below)
# 2. Smoke test (stratified subset, pipeline validation only)
python code/fedxai_ids_gpu_4workers.py --smoke --worker-id 0 --data-dir Dataset

# 3. Full run, one worker per dataset
#    0 = ML-EdgeIIoT (Edge-IIoTset), 1 = CIC-IoT2023, 2 = UNSW-NB15, 3 = Custom-Attack-Suite
python code/fedxai_ids_gpu_4workers.py --worker-id 0 --data-dir Dataset --device cuda
python code/fedxai_ids_gpu_4workers.py --worker-id 1 --data-dir Dataset --device cuda
python code/fedxai_ids_gpu_4workers.py --worker-id 2 --data-dir Dataset --device cuda
python code/fedxai_ids_gpu_4workers.py --worker-id 3 --data-dir Dataset --device cuda
```

Defaults match the manuscript: 10 seeds, 20 clients, 30 rounds, 5 local epochs, α = 0.5, μ = 0.01. Each worker writes to its own output directory (`supercomputer_outputs/` by default); smoke-test outputs must not be mixed with production results.

**Datasets (not redistributed).** Expected files under `Dataset/`: `ML-EdgeIIoT-dataset.csv`, `ciciot23.csv`, `UNSW_NB15_training-set.parquet`, `UNSW_NB15_testing-set.parquet`, and the 13 Custom-Attack-Suite CSV files. Obtain CIC-IoT2023, Edge-IIoTset and UNSW-NB15 from their original sources. Custom-Attack-Suite is in-house data.

## Notes on the reported results

- Values are descriptive means ± standard deviations over seeds; no statistical significance, formal privacy guarantee, or convergence guarantee is claimed.
- Tables and figures here are exactly as they appear in the manuscript draft of 29 October. The draft notes that some F1 values (e.g. CIC-IoT2023 FedAvg-IID) differ from the abstract and response-to-reviewers file; regenerate all tables from the single verified results registry before the final release.
- Table 13 compares studies that use different datasets, preprocessing and protocols; it is illustrative, not a head-to-head benchmark.

## Citation

See `CITATION.cff`, or cite the Zenodo record once the DOI is issued.
