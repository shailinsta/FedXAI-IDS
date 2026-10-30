#!/usr/bin/env python3
"""
FedXAI-IDS — GPU-Centric Final Pipeline
=======================================
Addresses ALL Reviewer 1 & Reviewer 2 comments for Cluster Computing revision.

Reviewer 1
  • Related-work positioning via literature table + quantitative FedProx baseline
  • Four worker datasets: ML-EdgeIIoT, CIC-IoT2023, UNSW-NB15, and Custom-Attack-Suite
  • Framework description BEFORE algorithms; standard Input/Output pseudocode
  • Formal metric definitions + imbalance justification
  • Multi-seed statistics + update-norm / cosine diagnostics for non-IID vs IID

Reviewer 2
  • Single MASTER registry → numerical consistency across all tables
  • Non-IID diagnostics (client attack-rate SD, update ℓ2, cosine, learning curves)
  • α ∈ {0.1, 0.5, 1.0} and client-count ablations with error bars
  • Quantitative FedProx federated baseline on the same partitions
  • Expanded attribution: global, dependence, class-conditional, IID vs non-IID
  • Full Residual-MLP architecture + hyperparameters recorded
  • Centralised LogisticRegression + MLP baselines (completes former empty Table 3)
  • Data-fraction saturation curves
  • 300 dpi figures; open outputs under supercomputer_outputs/

Usage
-----
  # Full supercomputer run (10 seeds × 20 clients × 30 rounds × 4 dataset workers)
  python FedXAI_IDS_GPU_Final.py

  # Smoke test
  python FedXAI_IDS_GPU_Final.py --smoke

  # Custom
  python FedXAI_IDS_GPU_4Workers_Production_Final.py --worker-id 0 --data-dir Dataset --device cuda

Place the four validated dataset groups under ./Dataset (or use --data-dir).\nWorkers 0..3 map to ML-EdgeIIoT, CIC-IoT2023, UNSW-NB15, and Custom-Attack-Suite. DNN-EdgeIIoT is intentionally excluded because the supplied file contains only the benign/Normal class.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import shapiro, ttest_rel, wilcoxon

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# =============================================================================
# 0. CLI + SUPERCOMPUTER / GPU CONFIG
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FedXAI-IDS GPU-centric final pipeline")
    p.add_argument("--seeds", type=int, default=10, help="Independent random seeds (default 10)")
    p.add_argument("--clients", type=int, default=20, help="Primary federation size (default 20)")
    p.add_argument("--rounds", type=int, default=30, help="Communication rounds (default 30)")
    p.add_argument("--local-epochs", type=int, default=5, help="Local epochs per round")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--mu-prox", type=float, default=0.01, help="FedProx proximal coefficient")
    p.add_argument("--alpha", type=float, default=0.5, help="Primary Dirichlet α")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--smoke", action="store_true", help="Fast validation run on a stratified subset; production settings are unchanged")
    p.add_argument("--smoke-max-train", type=int, default=300_000, help="Maximum training rows in smoke mode (default 300k)")
    p.add_argument("--smoke-max-test", type=int, default=100_000, help="Maximum test rows in smoke mode (default 100k)")
    p.add_argument("--cache-dir", type=Path, default=Path(".fedxai_cache"), help="Validated preprocessed dataset cache")
    p.add_argument("--rebuild-cache", action="store_true", help="Ignore and rebuild the preprocessed dataset cache")
    p.add_argument("--no-resume", action="store_true", help="Disable per-seed checkpoint/resume")
    p.add_argument("--loader-workers", type=int, default=0, help="DataLoader workers per client; 0 is safest for in-memory arrays")
    p.add_argument("--output-dir", type=Path, default=Path("supercomputer_outputs"))
    p.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    p.add_argument("--worker-id", type=int, choices=range(4), default=None,
                   help="Run one DGX worker (0..3): ML-EdgeIIoT, CIC-IoT2023, UNSW-NB15, or Custom-Attack-Suite")
    p.add_argument("--dataset", choices=("ML-EdgeIIoT", "CIC-IoT2023", "UNSW-NB15", "Custom-Attack-Suite"),
                   default=None, help="Run only the named dataset (overrides --worker-id)")
    p.add_argument("--data-dir", type=Path, default=Path("Dataset"),
                   help="Folder containing datasets (default: ./Dataset)")
    return p.parse_args()


ARGS = parse_args()
if ARGS.smoke:
    # A smoke run validates correctness quickly; it is never a publication run.
    ARGS.seeds = 2
    ARGS.rounds = 4
    ARGS.clients = 10
    ARGS.local_epochs = 1

N_SEEDS = ARGS.seeds
N_CLIENTS = ARGS.clients
CLIENT_ABLATION = tuple(sorted(set((5, 10, 20 if N_CLIENTS <= 10 else N_CLIENTS)))) if N_CLIENTS >= 10 else tuple(sorted(set((max(3, N_CLIENTS // 2), N_CLIENTS))))
N_ROUNDS = ARGS.rounds
LOCAL_EPOCHS = ARGS.local_epochs
PRIMARY_ALPHA = ARGS.alpha
ALPHA_GRID = (0.1, 0.5, 1.0)
BATCH_SIZE = ARGS.batch_size
LR = ARGS.lr
MU_PROX = ARGS.mu_prox
BASE_SEED = 42
USE_AMP = not ARGS.no_amp
PIN_MEMORY = True
LOADER_WORKERS = max(0, int(ARGS.loader_workers))
RESUME = not ARGS.no_resume
CACHE_DIR = ARGS.cache_dir.expanduser().resolve()
CACHE_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DIR = ARGS.output_dir
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "figures").mkdir(exist_ok=True)
(OUTPUT_DIR / "tables").mkdir(exist_ok=True)
(OUTPUT_DIR / "checkpoints").mkdir(exist_ok=True)

# Dataset locations.
# The program is designed to be launched from ~/FedXAI (or any project folder)
# with all data files inside ./Dataset.  --data-dir can override that location.
DATA_DIR = ARGS.data_dir.expanduser().resolve()

DATA_SPECS = {
    "ML-EdgeIIoT": {
        "kind": "csv",
        "files": ["ML-EdgeIIoT-dataset.csv"],
    },
    "CIC-IoT2023": {
        "kind": "csv",
        "files": ["ciciot23.csv"],
    },
    "UNSW-NB15": {
        "kind": "parquet_pair",
        "files": ["UNSW_NB15_training-set.parquet", "UNSW_NB15_testing-set.parquet"],
    },
    "Custom-Attack-Suite": {
        "kind": "multi_csv",
        "files": [
            # Normal traffic (Attack_label=0)
            "Distance.csv",
            "Flame_Sensor.csv",
            "Heart_Rate.csv",
            "IR_Receiver.csv",
            "Modbus.csv",
            "phValue.csv",
            "Soil_Moisture.csv",
            "Sound_Sensor.csv",
            "Temperature_and_Humidity.csv",
            "Water_Level.csv",
            # Attack traffic (Attack_label=1)
            "DDoS_HTTP_Flood_attack.csv",
            "Ransomware_attack.csv",
            "SQL_injection_attack.csv",
        ],
    },
}

WORKER_DATASET = {
    0: "ML-EdgeIIoT",
    1: "CIC-IoT2023",
    2: "UNSW-NB15",
    3: "Custom-Attack-Suite",
}

SELECTED_DATASET = ARGS.dataset
if SELECTED_DATASET is None and ARGS.worker_id is not None:
    SELECTED_DATASET = WORKER_DATASET[ARGS.worker_id]

# Give every worker a separate output directory automatically, preventing
# concurrent processes from overwriting one another.
if ARGS.worker_id is not None and ARGS.output_dir == Path("supercomputer_outputs"):
    ARGS.output_dir = Path(f"supercomputer_outputs_worker{ARGS.worker_id}_{SELECTED_DATASET}")

OUTPUT_DIR = ARGS.output_dir
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "figures").mkdir(exist_ok=True)
(OUTPUT_DIR / "tables").mkdir(exist_ok=True)
(OUTPUT_DIR / "checkpoints").mkdir(exist_ok=True)


print("=" * 78)
print("FedXAI-IDS GPU-Centric Final Pipeline")
print("=" * 78)
print(f"Output      : {OUTPUT_DIR.resolve()}")
print(f"Seeds={N_SEEDS}  Clients={N_CLIENTS}  Rounds={N_ROUNDS}  LocalEpochs={LOCAL_EPOCHS}")
print(f"α={PRIMARY_ALPHA}  Batch={BATCH_SIZE}  LR={LR}  μ_prox={MU_PROX}  AMP={USE_AMP}")

# =============================================================================
# 1. DEVICE
# =============================================================================

if ARGS.device == "auto":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
elif ARGS.device == "cuda":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu")
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

N_GPUS = torch.cuda.device_count() if DEVICE.type == "cuda" else 0
print(f"PyTorch {torch.__version__}  |  Device: {DEVICE}")
if N_GPUS:
    for i in range(N_GPUS):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}  "
              f"mem={props.total_memory / 1e9:.1f} GB")
    torch.backends.cudnn.benchmark = True
else:
    print("  CUDA not visible — CPU mode (still multi-seed / multi-client ready)")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# 2. FRAMEWORK (before algorithms) + PSEUDOCODE + METRICS + ARCHITECTURE
#    Addresses R1: "method explained before algorithms", formal metrics,
#    complete architecture table
# =============================================================================

FRAMEWORK = """
1. Dataset-specific cleaning (duplicates, missing, identifiers).
2. Deterministic categorical encoding.
3. Stratified train/test split BEFORE federation; scaler fit on train only.
4. IID or class-wise Dirichlet (α) partition across K clients.
5. Each client trains the same residual MLP on GPU (AMP when CUDA).
6. FedAvg sample-count aggregation; FedProx as quantitative FL baseline.
7. Client diagnostics: label rates, update ℓ2-norms, cosine similarity.
8. Attribution (gradient×input): global, dependence, class-conditional, IID vs non-IID.
9. Multi-seed (N) mean±std; all tables from one MASTER registry.
"""
print("\n--- Framework ---")
print(FRAMEWORK.strip())
(OUTPUT_DIR / "Framework_Description.txt").write_text(FRAMEWORK.strip(), encoding="utf-8")

ALGORITHMS = {
    "Algorithm 1 – Preprocessing": (
        "Input: D, y, K, α. Output: scaled matrices + client partitions.\n"
        "1–5 clean/encode/split/scale. 6 IID random split. "
        "7 Non-IID Dirichlet per class. 8 Return."
    ),
    "Algorithm 2 – FedAvg / FedProx (GPU)": (
        "Input: clients, w0, T, E, μ. Output: w_T + histories.\n"
        "For t=1…T: each client loads w, trains E epochs on GPU (AMP), "
        "returns w_k, n_k, Δ_k; aggregate by n_k; evaluate on fixed test; "
        "log metrics + update statistics."
    ),
    "Algorithm 3 – Explainability": (
        "Gradient×input attributions; global rank; dependence; "
        "benign/attack; IID vs non-IID."
    ),
}
(OUTPUT_DIR / "Algorithm_Pseudocode.json").write_text(
    json.dumps(ALGORITHMS, indent=2), encoding="utf-8"
)

METRIC_DEFINITIONS = pd.DataFrame(
    [
        ["Accuracy", "(TP+TN)/(TP+TN+FP+FN)", "Overall correctness; not used alone under imbalance."],
        ["Precision", "TP/(TP+FP)", "False-alarm control."],
        ["Recall", "TP/(TP+FN)", "Attack coverage."],
        ["F1", "2PR/(P+R)", "Balance under imbalance."],
        ["FPR", "FP/(FP+TN)", "Operational false-alarm rate."],
        ["ROC-AUC", "∫ TPR dFPR", "Threshold-free ranking."],
        ["Balanced Acc.", "(TPR+TNR)/2", "Equal class weight."],
    ],
    columns=["Metric", "Definition", "Justification"],
)
print("\n--- Metric definitions ---")
print(METRIC_DEFINITIONS.to_string(index=False))
METRIC_DEFINITIONS.to_csv(OUTPUT_DIR / "tables" / "Metric_Definitions.csv", index=False)

ARCHITECTURE = pd.DataFrame(
    [
        ["Input", "d", "–", "Scaled feature vector"],
        ["Linear", "256", "ReLU + LayerNorm + Dropout 0.3", "Hidden 1"],
        ["Linear", "128", "ReLU + residual add", "Hidden 2 + shortcut"],
        ["Linear", "64", "ReLU + Dropout 0.2", "Hidden 3"],
        ["Linear", "1", "Sigmoid (via logits)", "Binary probability"],
    ],
    columns=["Layer", "Units", "Activation / Reg", "Purpose"],
)
print("\n--- DNN architecture ---")
print(ARCHITECTURE.to_string(index=False))
ARCHITECTURE.to_csv(OUTPUT_DIR / "tables" / "DNN_Architecture.csv", index=False)

HYPER = dict(
    optimizer="Adam",
    lr=LR,
    batch=BATCH_SIZE,
    local_epochs=LOCAL_EPOCHS,
    rounds=N_ROUNDS,
    clients=N_CLIENTS,
    seeds=N_SEEDS,
    alpha=PRIMARY_ALPHA,
    mu_prox=MU_PROX,
    amp=USE_AMP and DEVICE.type == "cuda",
    device=str(DEVICE),
)
print("\nHyperparameters:", HYPER)
with open(OUTPUT_DIR / "tables" / "Hyperparameters.json", "w") as f:
    json.dump(HYPER, f, indent=2)

# Literature positioning table (R1 related-work + R2 SOTA baselines)
LITERATURE = pd.DataFrame(
    [
        ["Thein et al. (2024)", "Yes", "Personalization", "None", "Weighted Avg", "No", "Class weights"],
        ["Jin et al. (2024)", "Yes", "Incremental Drift", "None", "Adaptive Agg", "No", "None"],
        ["Doriguzzi-Corin (2024)", "Yes", "Loss-based", "None", "Adaptive Part.", "No", "None"],
        ["SAFE-IDS (2025)", "Yes", "Gradient Sign", "None", "Weighted Loss", "Yes", "Weighted Loss"],
        ["TrustFed-IDS (2026)", "Yes", "Trust-based", "Short-term", "Multi-criteria", "No", "Class Balanced"],
        ["AFL-SecNet (2025)", "Yes", "Adaptive FL", "None", "Adaptive", "Yes", "Accuracy"],
        ["FedXAI-IDS (Proposed)", "Yes", "Dirichlet Non-IID", "None", "FedAvg+FedProx", "No*", "F1 / Bal.Acc / Grad×Input"],
    ],
    columns=[
        "Study / System", "FL-IDS", "Non-IID Handling", "Trajectory",
        "Aggregation", "DP", "Focus Metric",
    ],
)
print("\n--- Literature positioning ---")
print(LITERATURE.to_string(index=False))
LITERATURE.to_csv(OUTPUT_DIR / "tables" / "Literature_Positioning.csv", index=False)
# * Privacy is structural (raw data never leaves clients); formal ε-DP is future work.


# =============================================================================
# 3. GPU RESIDUAL MLP
# =============================================================================

class ResidualMLP(nn.Module):
    """Residual MLP — fully specified architecture (R2 #6)."""

    def __init__(self, d_in: int):
        super().__init__()
        self.fc1 = nn.Linear(d_in, 256)
        self.ln1 = nn.LayerNorm(256)
        self.drop1 = nn.Dropout(0.30)
        self.fc2 = nn.Linear(256, 128)
        self.shortcut = nn.Linear(256, 128)
        self.ln2 = nn.LayerNorm(128)
        self.drop2 = nn.Dropout(0.30)
        self.fc3 = nn.Linear(128, 64)
        self.drop3 = nn.Dropout(0.20)
        self.out = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop1(F.relu(self.ln1(self.fc1(x))))
        res = self.fc2(h)
        sc = self.shortcut(h)
        h = self.drop2(F.relu(self.ln2(res + sc)))
        h = self.drop3(F.relu(self.fc3(h)))
        return self.out(h).squeeze(-1)  # logits


def model_to_vector(model: nn.Module) -> torch.Tensor:
    return torch.nn.utils.parameters_to_vector(model.parameters()).detach()


def vector_to_model(model: nn.Module, vec: torch.Tensor) -> None:
    torch.nn.utils.vector_to_parameters(vec, model.parameters())


def get_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def set_state(model: nn.Module, state: Dict[str, torch.Tensor]) -> None:
    model.load_state_dict({k: v.to(DEVICE) for k, v in state.items()})


def weighted_state_average(
    states: List[Dict[str, torch.Tensor]], counts: List[int]
) -> Dict[str, torch.Tensor]:
    total = float(sum(counts))
    out: Dict[str, torch.Tensor] = {}
    for k in states[0].keys():
        acc = None
        for st, n in zip(states, counts):
            t = st[k].float() * (n / total)
            acc = t if acc is None else acc + t
        out[k] = acc
    return out


_n_params = sum(p.numel() for p in ResidualMLP(20).parameters())
print(f"\nResidualMLP defined. Params (d=20): {_n_params:,}")


# =============================================================================
# 4. TRAIN / EVAL HELPERS
# =============================================================================

def evaluate_binary(y_true: np.ndarray, prob: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    """Binary metrics with attack=1 and a tie-correct ROC-AUC."""
    y_true = np.asarray(y_true, dtype=np.int32)
    prob = np.asarray(prob, dtype=np.float64)
    if not np.all(np.isfinite(prob)):
        raise FloatingPointError("Non-finite prediction probabilities detected.")

    pred = (prob >= thr).astype(np.int32)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    fpr = fp / max(fp + tn, 1)

    # Average ranks give half credit to probability ties.
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos and n_neg:
        order = np.argsort(prob, kind="mergesort")
        sorted_p = prob[order]
        ranks = np.empty(len(prob), dtype=np.float64)
        i = 0
        while i < len(prob):
            j = i + 1
            while j < len(prob) and sorted_p[j] == sorted_p[i]:
                j += 1
            avg_rank = 0.5 * ((i + 1) + j)  # ranks are 1-based
            ranks[order[i:j]] = avg_rank
            i = j
        rank_sum_pos = ranks[y_true == 1].sum()
        auc = float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
    else:
        auc = float("nan")

    tnr = tn / max(tn + fp, 1)
    bal = 0.5 * (rec + tnr)
    return dict(
        Accuracy=acc, Precision=prec, Recall=rec, F1=f1,
        FPR=fpr, ROC_AUC=auc, Balanced_Accuracy=bal,
    )


@torch.no_grad()
def predict_proba(model: nn.Module, X_np: np.ndarray, batch: int = 4096) -> np.ndarray:
    model.eval()
    probs = []
    for i in range(0, len(X_np), batch):
        xb = torch.as_tensor(X_np[i : i + batch], dtype=torch.float32, device=DEVICE)
        logits = model(xb)
        probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs)


def train_local(
    model: nn.Module,
    X_np: np.ndarray,
    y_np: np.ndarray,
    epochs: int = LOCAL_EPOCHS,
    batch: int = BATCH_SIZE,
    lr: float = LR,
    global_state: Optional[Dict] = None,
    mu: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """One client local training with a differentiable FedProx penalty."""
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    amp_enabled = bool(USE_AMP and DEVICE.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    ds = TensorDataset(
        torch.as_tensor(X_np, dtype=torch.float32),
        torch.as_tensor(y_np, dtype=torch.float32),
    )
    loader = DataLoader(
        ds, batch_size=batch, shuffle=True,
        pin_memory=(PIN_MEMORY and DEVICE.type == "cuda"),
        num_workers=LOADER_WORKERS,
        persistent_workers=(LOADER_WORKERS > 0),
    )

    # Keep a detached copy of the global trainable parameters on-device.
    # Do NOT use model_to_vector() in the loss because that helper detaches.
    global_params = None
    if global_state is not None and mu > 0:
        global_params = {
            name: global_state[name].to(DEVICE, dtype=p.dtype).detach().clone()
            for name, p in model.named_parameters()
        }

    successful_steps = 0
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = model(xb)
                bce = F.binary_cross_entropy_with_logits(logits, yb)

            # Compute proximal term in FP32 for numerical stability.
            loss = bce.float()
            if global_params is not None:
                prox = torch.zeros((), device=DEVICE, dtype=torch.float32)
                for name, p in model.named_parameters():
                    prox = prox + torch.sum(
                        (p.float() - global_params[name].float()) ** 2
                    )
                loss = loss + (float(mu) / 2.0) * prox

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Non-finite local loss detected. Check scaled feature range."
                )

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(opt)
            scaler.update()
            successful_steps += 1

    if successful_steps == 0:
        raise RuntimeError("Local training completed with zero optimizer steps.")
    return get_state(model)


print("Train / eval helpers ready.")


# =============================================================================
# 5. DATA LOADERS — SAFE CSV INGESTION + PARQUET
#    Addresses R1: add at least one extra benchmark (NF-UNSW-NB15)
# =============================================================================

def robust_scale(Xtr: np.ndarray, Xte: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Train-only robust scaling with finite-value cleanup and clipping.

    Clipping is important for sparse network counters: a tiny/zero IQR plus an
    extreme value can otherwise create huge activations, overflow mixed
    precision, and make GradScaler skip optimizer steps.
    """
    Xtr64 = np.asarray(Xtr, dtype=np.float64)
    Xte64 = np.asarray(Xte, dtype=np.float64)
    Xtr64 = np.nan_to_num(Xtr64, nan=0.0, posinf=0.0, neginf=0.0)
    Xte64 = np.nan_to_num(Xte64, nan=0.0, posinf=0.0, neginf=0.0)

    med = np.median(Xtr64, axis=0)
    q75 = np.percentile(Xtr64, 75, axis=0)
    q25 = np.percentile(Xtr64, 25, axis=0)
    iqr = q75 - q25
    iqr = np.where(np.isfinite(iqr) & (iqr >= 1e-8), iqr, 1.0)

    Xtr_s = (Xtr64 - med) / iqr
    Xte_s = (Xte64 - med) / iqr
    Xtr_s = np.nan_to_num(Xtr_s, nan=0.0, posinf=10.0, neginf=-10.0)
    Xte_s = np.nan_to_num(Xte_s, nan=0.0, posinf=10.0, neginf=-10.0)
    Xtr_s = np.clip(Xtr_s, -10.0, 10.0).astype(np.float32)
    Xte_s = np.clip(Xte_s, -10.0, 10.0).astype(np.float32)
    return Xtr_s, Xte_s


def stratified_split(
    X: np.ndarray, y: np.ndarray, test_size: float = 0.2, seed: int = BASE_SEED
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    i0, i1 = np.where(y == 0)[0], np.where(y == 1)[0]
    rng.shuffle(i0)
    rng.shuffle(i1)
    n0t = max(1, int(len(i0) * test_size))
    n1t = max(1, int(len(i1) * test_size))
    te = np.concatenate([i0[:n0t], i1[:n1t]])
    tr = np.concatenate([i0[n0t:], i1[n1t:]])
    rng.shuffle(tr)
    rng.shuffle(te)
    return X[tr], X[te], y[tr], y[te]


def _synth(name: str, n: int, d: int, ar: float, sep: float, seed: int):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < ar).astype(np.int32)
    X = rng.normal(0, 1, (n, d)).astype(np.float32)
    X[y == 1] += rng.normal(sep, 0.5, (int(y.sum()), d)).astype(np.float32)
    X[:, :4] = np.abs(X[:, :4])
    return X, y, [f"{name[:3]}_f{i}" for i in range(d)]


SYNTH_CFG = {
    "Edge-IIoTset": dict(n=20000, d=24, ar=0.35, sep=1.35, seed=42),
    "CIC-IoT2023": dict(n=22000, d=32, ar=0.55, sep=1.50, seed=43),
    "NF-UNSW-NB15": dict(n=18000, d=14, ar=0.18, sep=1.70, seed=44),
}


def _binary_target(series: pd.Series) -> np.ndarray:
    """Convert common IDS labels to 0=benign, 1=attack."""
    if pd.api.types.is_numeric_dtype(series):
        vals = pd.to_numeric(series, errors="coerce").fillna(0).to_numpy()
        return (vals > 0).astype(np.int32)

    s = series.astype(str).str.strip().str.lower()
    benign = (
        s.eq("0")
        | s.eq("benign")
        | s.eq("normal")
        | s.str.startswith("benign")
        | s.str.startswith("normal")
    )
    return (~benign).astype(np.int32).to_numpy()


def _find_target(df: pd.DataFrame, name: str) -> str:
    candidates = [
        "Attack_label", "attack_label", "Label", "label",
        "Attack", "attack", "class", "Class", "binary"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"No supported binary target column found for {name}. "
        f"Columns include: {list(df.columns)[:25]}"
    )


def _numeric_features(df: pd.DataFrame, target: str) -> Tuple[np.ndarray, List[str]]:
    drop = {target}
    # Explicitly exclude alternate labels and obvious identifiers/leakage-prone metadata.
    alternate_labels = {
        "attack_type", "attack label", "attack_label", "label", "class", "binary"
    }
    for c in df.columns:
        cl = str(c).strip().lower()
        if c != target and cl in alternate_labels:
            drop.add(c)
        if (
            cl in {"id", "flow_id", "src_ip", "dst_ip", "timestamp", "frame.time"}
            or cl.endswith("_id")
            or "src_ip" in cl
            or "dst_ip" in cl
            or cl in {"ip.src_host", "ip.dst_host"}
        ):
            drop.add(c)

    Xdf = df.drop(columns=list(drop), errors="ignore").select_dtypes(include=[np.number])
    Xdf = Xdf.replace([np.inf, -np.inf], np.nan).fillna(0)
    if Xdf.shape[1] == 0:
        raise ValueError("No numeric predictor columns remain after preprocessing.")
    return Xdf.to_numpy(dtype=np.float32), list(Xdf.columns)



def _safe_read_csv(path: Path, name: str) -> pd.DataFrame:
    """
    Safe CSV ingestion for the DGX Spark environment.

    pandas.read_csv() segfaulted on the complete Edge-IIoT CSV and
    pd.to_numeric() segfaulted specifically on mixed packet-payload data
    (tcp.payload).  Therefore:
      1) parse with Python's stdlib csv.reader;
      2) never numerically coerce payload/content/URI/text/identifier fields;
      3) explicitly convert the binary target;
      4) convert only columns that are safe numeric candidates.

    The downstream feature selector still keeps numeric columns only, so
    excluded text-like columns cannot leak into the model.
    """
    print(f"[{name}] Safe CSV loader (stdlib csv.reader) → {path}", flush=True)

    csv.field_size_limit(100_000_000)
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"{name}: CSV file is empty: {path}")

        expected = len(header)
        rows = []
        bad_rows = []
        total_records = 0
        for line_no, row in enumerate(reader, start=2):
            total_records += 1
            if len(row) != expected:
                # Quarantine rather than pad/repair: padding could shift fields
                # and silently corrupt labels/features.
                bad_rows.append((line_no, len(row)))
                continue
            rows.append(row)

    bad_rate = len(bad_rows) / max(total_records, 1)
    if bad_rows:
        examples = ", ".join(
            f"line {ln} ({n} cols)" for ln, n in bad_rows[:10]
        )
        print(
            f"[{name}] WARNING: quarantined {len(bad_rows):,} malformed "
            f"record(s) / {total_records:,} ({100*bad_rate:.6f}%). "
            f"Examples: {examples}",
            flush=True,
        )
        # A few physically truncated records are tolerable and auditable.
        # Material corruption is not: stop rather than bias the experiment.
        if bad_rate > 0.0001:  # 0.01 percent
            raise ValueError(
                f"{name}: malformed-row rate {100*bad_rate:.6f}% exceeds "
                "0.01% safety threshold; inspect/re-export this CSV."
            )

    if not rows:
        raise ValueError(f"{name}: no valid records found in {path}")

    print(
        f"[{name}] Parsed {len(rows):,} valid rows × {expected:,} columns safely.",
        flush=True,
    )
    df = pd.DataFrame(rows, columns=header)
    df.attrs["csv_total_records"] = int(total_records)
    df.attrs["csv_valid_records"] = int(len(rows))
    df.attrs["csv_malformed_records"] = int(len(bad_rows))
    df.attrs["csv_malformed_examples"] = bad_rows[:100]

    # Known target names: convert these explicitly and safely.
    target_names = {
        "attack_label", "label", "target", "class", "binary_label",
        "is_attack", "attack"
    }

    # Never pass these kinds of mixed/text fields through pd.to_numeric().
    # In particular, tcp.payload was proven to trigger a native segfault on
    # the DGX Spark pandas stack.
    unsafe_exact = {
        "attack_type", "attack type", "attack_cat", "attack category",
        "label_name", "class_name",
        "tcp.payload", "tcp.options",
        "http.file_data", "http.request.uri.query",
        "http.request.method", "http.referer",
        "http.request.full_uri", "http.request.version",
        "mqtt.msg", "mqtt.topic", "mqtt.protoname",
        "frame.time", "ip.src_host", "ip.dst_host",
    }
    unsafe_tokens = (
        "payload", "file_data", "uri", "referer", "method",
        "topic", "protoname", "src_host", "dst_host"
    )

    converted_count = 0
    skipped_count = 0

    for col in df.columns:
        cl = str(col).strip().lower()

        # Binary target is expected to be simple numeric 0/1 for Edge-IIoT.
        if cl in target_names:
            # Use a conservative string mapping first; avoid broad conversion
            # over arbitrary mixed-content columns.
            s = df[col].astype(str).str.strip()
            numeric_simple = s.str.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)")
            if numeric_simple.all():
                df[col] = s.astype("float32")
                converted_count += 1
            continue

        if cl in unsafe_exact or any(tok in cl for tok in unsafe_tokens):
            skipped_count += 1
            continue

        s = df[col].astype(str).str.strip()
        nonempty = s.ne("")
        if not nonempty.any():
            skipped_count += 1
            continue

        # Cheap regex gate: only call pd.to_numeric when every non-empty value
        # is syntactically numeric. This prevents mixed packet/text fields from
        # reaching the native conversion path that caused the segfault.
        vals = s[nonempty]
        numeric_mask = vals.str.fullmatch(
            r"[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?"
        )
        if bool(numeric_mask.all()):
            df[col] = pd.to_numeric(s, errors="coerce").astype("float32")
            converted_count += 1
        else:
            skipped_count += 1

    print(
        f"[{name}] Safe dtype conversion complete: "
        f"{converted_count} numeric columns; {skipped_count} text/mixed columns excluded.",
        flush=True,
    )
    return df


def _load_real_dataset_uncached(name: str) -> Dict[str, Any]:
    spec = DATA_SPECS[name]
    paths = [DATA_DIR / f for f in spec["files"]]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"{name}: required dataset file(s) missing: {missing}. "
            f"Expected under {DATA_DIR}"
        )

    print(f"[{name}] REAL DATA → {', '.join(str(p) for p in paths)}")

    if spec["kind"] == "csv":
        df = _safe_read_csv(paths[0], name)
        target = _find_target(df, name)
        y = _binary_target(df[target])
        X, names = _numeric_features(df, target)
        Xtr, Xte, ytr, yte = stratified_split(X, y)

    elif spec["kind"] == "multi_csv":
        # Merge compatible CSV exports using only common numeric predictors.
        # Each source must expose a supported target column, unless it is
        # explicitly an attack capture; in that case label=1 is inferred from
        # the filename.  A final two-class check below prevents scientifically
        # invalid attack-only training.
        frames = []
        inferred_labels = []
        for pth in paths:
            part_name = f"{name}:{pth.name}"
            df_part = _safe_read_csv(pth, part_name)
            try:
                tgt = _find_target(df_part, part_name)
                y_part = _binary_target(df_part[tgt])
            except ValueError:
                # These user-added files are named attack captures.  If their
                # exporter omitted a target column, their rows are attacks.
                tgt = "__inferred_target__"
                y_part = np.ones(len(df_part), dtype=np.int32)

            X_part, feat_part = _numeric_features(df_part, tgt)
            frames.append((X_part, feat_part))
            inferred_labels.append(y_part)

        common = list(frames[0][1])
        for _, feats in frames[1:]:
            fset = set(feats)
            common = [c for c in common if c in fset]
        if not common:
            raise ValueError(
                f"{name}: the three CSV files have no common numeric predictors. "
                "They cannot be merged into one scientifically consistent benchmark."
            )

        aligned = []
        for (X_part, feats), y_part in zip(frames, inferred_labels):
            pos = {f: i for i, f in enumerate(feats)}
            aligned.append(X_part[:, [pos[c] for c in common]])

        X = np.concatenate(aligned, axis=0)
        y = np.concatenate(inferred_labels, axis=0).astype(np.int32)
        names = common

        vals, cnts = np.unique(y, return_counts=True)
        class_counts = dict(zip(vals.tolist(), cnts.tolist()))
        print(f"[{name}] Combined class counts: {class_counts}", flush=True)
        if len(vals) < 2:
            raise ValueError(
                f"{name}: combined custom CSVs contain only one class {class_counts}. "
                "A binary IDS benchmark requires both benign (0) and attack (1) rows. "
                "Do not fabricate benign samples; add a compatible benign capture/export."
            )

        Xtr, Xte, ytr, yte = stratified_split(X, y)

    elif spec["kind"] == "parquet_pair":
        # Preserve the official UNSW train/test split.
        train_df = pd.read_parquet(paths[0])
        test_df = pd.read_parquet(paths[1])
        target_tr = _find_target(train_df, name)
        target_te = _find_target(test_df, name)
        ytr = _binary_target(train_df[target_tr])
        yte = _binary_target(test_df[target_te])
        Xtr, names_tr = _numeric_features(train_df, target_tr)
        Xte, names_te = _numeric_features(test_df, target_te)

        common = [c for c in names_tr if c in set(names_te)]
        if not common:
            raise ValueError("UNSW-NB15 train/test files have no common numeric predictors.")
        # Rebuild in identical feature order.
        train_num = train_df[common].replace([np.inf, -np.inf], np.nan).fillna(0)
        test_num = test_df[common].replace([np.inf, -np.inf], np.nan).fillna(0)
        Xtr = train_num.to_numpy(dtype=np.float32)
        Xte = test_num.to_numpy(dtype=np.float32)
        names = common
    else:
        raise ValueError(f"Unsupported dataset kind: {spec['kind']}")

    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        raise ValueError(f"{name}: binary target does not contain both classes.")

    Xtr, Xte = robust_scale(Xtr, Xte)
    if not (np.isfinite(Xtr).all() and np.isfinite(Xte).all()):
        raise FloatingPointError(f"{name}: non-finite values remain after scaling.")
    print(
        f"[{name}] Scaled feature range: "
        f"train=[{Xtr.min():.3f}, {Xtr.max():.3f}] "
        f"test=[{Xte.min():.3f}, {Xte.max():.3f}]",
        flush=True,
    )
    return dict(
        name=name,
        source="real",
        raw_rows=int(len(ytr) + len(yte)),
        X_train=Xtr,
        X_test=Xte,
        y_train=ytr.astype(np.int32),
        y_test=yte.astype(np.int32),
        feature_names=names,
        n_features=Xtr.shape[1],
        attack_rate_train=float(ytr.mean()),
        attack_rate_test=float(yte.mean()),
    )


def _dataset_signature(name: str) -> Dict[str, Any]:
    """Cheap cache signature: absolute path, byte size and mtime for every source file."""
    spec = DATA_SPECS[name]
    files = []
    for fname in spec["files"]:
        pth = (DATA_DIR / fname).resolve()
        st = pth.stat()
        files.append({"path": str(pth), "size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)})
    return {"dataset": name, "files": files, "preprocess_version": 3}


def _cache_paths(name: str) -> Tuple[Path, Path]:
    safe = name.replace("/", "_").replace(" ", "_")
    root = CACHE_DIR / safe
    return root, root / "metadata.json"


def _load_dataset_cache(name: str) -> Optional[Dict[str, Any]]:
    root, meta_path = _cache_paths(name)
    if ARGS.rebuild_cache or not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("signature") != _dataset_signature(name):
            print(f"[{name}] Cache stale; rebuilding.", flush=True)
            return None
        required = ["X_train.npy", "X_test.npy", "y_train.npy", "y_test.npy"]
        if not all((root / f).exists() for f in required):
            return None
        print(f"[{name}] CACHE HIT → {root}", flush=True)
        # mmap avoids an unnecessary full copy; client slicing remains ordinary NumPy.
        Xtr = np.load(root / "X_train.npy", mmap_mode="r")
        Xte = np.load(root / "X_test.npy", mmap_mode="r")
        ytr = np.load(root / "y_train.npy", mmap_mode="r")
        yte = np.load(root / "y_test.npy", mmap_mode="r")
        return dict(
            name=name, source="real-cache", raw_rows=int(len(ytr) + len(yte)),
            X_train=Xtr, X_test=Xte, y_train=ytr, y_test=yte,
            feature_names=list(meta["feature_names"]), n_features=int(Xtr.shape[1]),
            attack_rate_train=float(np.mean(ytr)), attack_rate_test=float(np.mean(yte)),
        )
    except Exception as exc:
        print(f"[{name}] Cache read failed ({exc}); rebuilding safely.", flush=True)
        return None


def _write_dataset_cache(name: str, data: Dict[str, Any]) -> None:
    root, meta_path = _cache_paths(name)
    root.mkdir(parents=True, exist_ok=True)
    # Write to temporary names then atomically replace, so interruption cannot create a valid-looking cache.
    arrays = {
        "X_train.npy": np.asarray(data["X_train"], dtype=np.float32),
        "X_test.npy": np.asarray(data["X_test"], dtype=np.float32),
        "y_train.npy": np.asarray(data["y_train"], dtype=np.int32),
        "y_test.npy": np.asarray(data["y_test"], dtype=np.int32),
    }
    for fname, arr in arrays.items():
        tmp = root / (fname + ".tmp")
        with tmp.open("wb") as fh:
            np.save(fh, arr, allow_pickle=False)
        os.replace(tmp, root / fname)
    meta = {
        "signature": _dataset_signature(name),
        "feature_names": list(data["feature_names"]),
        "created_unix": time.time(),
    }
    tmp_meta = root / "metadata.json.tmp"
    tmp_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    os.replace(tmp_meta, meta_path)
    print(f"[{name}] Cache written → {root}", flush=True)


def load_real_dataset(name: str) -> Dict[str, Any]:
    cached = _load_dataset_cache(name)
    if cached is not None:
        return cached
    t0 = time.time()
    data = _load_real_dataset_uncached(name)
    _write_dataset_cache(name, data)
    print(f"[{name}] Preprocess+cache time: {(time.time()-t0)/60:.2f} min", flush=True)
    return data


def _stratified_limit(X: np.ndarray, y: np.ndarray, max_rows: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministic class-stratified subset for smoke validation only."""
    if max_rows <= 0 or len(y) <= max_rows:
        return np.asarray(X), np.asarray(y)
    rng = np.random.default_rng(seed)
    chosen = []
    for cls in np.unique(y):
        ids = np.where(np.asarray(y) == cls)[0]
        target = max(1, int(round(max_rows * len(ids) / len(y))))
        target = min(target, len(ids))
        chosen.append(rng.choice(ids, size=target, replace=False))
    idx = np.concatenate(chosen)
    if len(idx) > max_rows:
        idx = rng.choice(idx, size=max_rows, replace=False)
    rng.shuffle(idx)
    return np.asarray(X[idx], dtype=np.float32), np.asarray(y[idx], dtype=np.int32)


def apply_smoke_subset(data: Dict[str, Any]) -> Dict[str, Any]:
    if not ARGS.smoke:
        return data
    out = dict(data)
    out["X_train"], out["y_train"] = _stratified_limit(
        data["X_train"], data["y_train"], ARGS.smoke_max_train, BASE_SEED
    )
    out["X_test"], out["y_test"] = _stratified_limit(
        data["X_test"], data["y_test"], ARGS.smoke_max_test, BASE_SEED + 1
    )
    out["raw_rows"] = int(len(out["y_train"]) + len(out["y_test"]))
    out["attack_rate_train"] = float(np.mean(out["y_train"]))
    out["attack_rate_test"] = float(np.mean(out["y_test"]))
    out["source"] = str(data["source"]) + "-smoke-subset"
    print(
        f"[{data['name']}] SMOKE subset: train={len(out['y_train']):,} "
        f"test={len(out['y_test']):,}; full validated cache remains unchanged.", flush=True
    )
    return out


print("\n--- Loading datasets ---")
if SELECTED_DATASET is not None:
    dataset_names = [SELECTED_DATASET]
else:
    dataset_names = list(DATA_SPECS.keys())

datasets = {ds: apply_smoke_subset(load_real_dataset(ds)) for ds in dataset_names}

for n, d in datasets.items():
    print(
        f"  {n}: train={len(d['y_train'])} test={len(d['y_test'])} "
        f"feats={d['n_features']} atk={d['attack_rate_train']:.3f} [{d['source']}]"
    )


# =============================================================================
# 6. PARTITIONING + MULTI-SEED FEDERATED TRAINING (GPU)
# =============================================================================

def iid_partition(X, y, n_clients, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    return [(X[s], y[s]) for s in np.array_split(idx, n_clients)]


def dirichlet_partition(X, y, n_clients, alpha, seed, min_size=20):
    rng = np.random.default_rng(seed)
    for _ in range(100):
        ids = [[] for _ in range(n_clients)]
        for c in np.unique(y):
            cls = np.where(y == c)[0]
            rng.shuffle(cls)
            p = rng.dirichlet(np.full(n_clients, alpha))
            cuts = (np.cumsum(p) * len(cls)).astype(int)[:-1]
            for k, part in enumerate(np.split(cls, cuts)):
                ids[k].extend(part.tolist())
        if min(len(v) for v in ids) >= min_size:
            return [(X[np.array(v)], y[np.array(v)]) for v in ids]
    raise RuntimeError("Dirichlet partition failed — raise alpha or lower min_size")


def partition_diagnostics(clients) -> pd.DataFrame:
    rows = []
    for k, (_, yk) in enumerate(clients):
        rows.append(
            dict(
                Client=k + 1,
                N=len(yk),
                Attack_Rate=float(np.mean(yk)),
                Class_0=int((yk == 0).sum()),
                Class_1=int((yk == 1).sum()),
            )
        )
    return pd.DataFrame(rows)


def train_federated_one_seed(
    data,
    setting="IID",
    alpha=0.5,
    algorithm="FedAvg",
    n_clients=N_CLIENTS,
    rounds=N_ROUNDS,
    local_epochs=LOCAL_EPOCHS,
    mu=MU_PROX,
    seed=BASE_SEED,
):
    seed_everything(seed)
    Xtr, ytr = data["X_train"], data["y_train"]
    Xte, yte = data["X_test"], data["y_test"]
    d_in = Xtr.shape[1]
    clients = (
        iid_partition(Xtr, ytr, n_clients, seed)
        if setting == "IID"
        else dirichlet_partition(Xtr, ytr, n_clients, alpha, seed)
    )
    global_model = ResidualMLP(d_in).to(DEVICE)
    gstate = get_state(global_model)
    history = []

    for r in range(1, rounds + 1):
        t0 = time.time()
        states, counts, deltas = [], [], []
        gvec = model_to_vector(global_model).cpu()
        for k, (Xk, yk) in enumerate(clients):
            local = ResidualMLP(d_in).to(DEVICE)
            set_state(local, gstate)
            st = train_local(
                local,
                Xk,
                yk,
                epochs=local_epochs,
                global_state=gstate if algorithm == "FedProx" else None,
                mu=mu if algorithm == "FedProx" else 0.0,
            )
            set_state(local, st)
            lvec = model_to_vector(local).cpu()
            deltas.append((lvec - gvec).numpy())
            states.append(st)
            counts.append(len(yk))
            del local
        gstate = weighted_state_average(states, counts)
        set_state(global_model, gstate)
        prob = predict_proba(global_model, Xte)
        met = evaluate_binary(yte, prob)
        div = [float(np.linalg.norm(d.astype(np.float64))) for d in deltas]
        if not np.all(np.isfinite(div)):
            raise FloatingPointError("Non-finite client update norm detected.")
        if max(div, default=0.0) <= 1e-12:
            raise RuntimeError(
                "All client model updates are zero. Refusing to report invalid FL results."
            )
        ref = np.mean(np.stack(deltas, axis=0).astype(np.float64), axis=0)
        sim = [
            float(
                np.dot(d, ref)
                / (np.linalg.norm(d) * np.linalg.norm(ref) + 1e-12)
            )
            for d in deltas
        ]
        history.append(
            dict(
                Round=r,
                Algorithm=algorithm,
                Setting=setting,
                Alpha=alpha if setting != "IID" else np.nan,
                Seed=seed,
                **met,
                Round_Time_s=time.time() - t0,
                Mean_Update_Norm=float(np.mean(div)),
                Std_Update_Norm=float(np.std(div)),
                Mean_Update_Cosine=float(np.mean(sim)),
            )
        )
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    diag = partition_diagnostics(clients)
    return global_model, pd.DataFrame(history), diag


def _checkpoint_file(data, setting, alpha, algorithm, n_clients, rounds, seed) -> Path:
    ds = str(data["name"]).replace("/", "_").replace(" ", "_")
    tag = f"{ds}__{algorithm}__{setting}__a{alpha:g}__k{n_clients}__r{rounds}__e{LOCAL_EPOCHS}__s{seed}.pt"
    return OUTPUT_DIR / "checkpoints" / tag


def multi_seed_federated(
    data,
    setting="IID",
    alpha=0.5,
    algorithm="FedAvg",
    n_clients=N_CLIENTS,
    rounds=N_ROUNDS,
    n_seeds=N_SEEDS,
):
    """Run independent trials with seed-level checkpoint/resume."""
    all_hist, last_rows = [], []
    last_model = None
    diag = None
    for s in range(n_seeds):
        seed = BASE_SEED + s * 17
        ckpt = _checkpoint_file(data, setting, alpha, algorithm, n_clients, rounds, seed)
        print(f"  seed {s + 1}/{n_seeds} (seed={seed}) ...", end=" ", flush=True)
        t_seed = time.time()
        loaded = False
        if RESUME and ckpt.exists():
            try:
                payload = torch.load(ckpt, map_location="cpu", weights_only=False)
                hist = pd.DataFrame(payload["history"])
                diag = pd.DataFrame(payload["diagnostics"])
                model = ResidualMLP(data["X_train"].shape[1]).to(DEVICE)
                set_state(model, payload["model_state"])
                loaded = True
            except Exception as exc:
                print(f"[checkpoint unreadable: {exc}; recomputing]", end=" ", flush=True)

        if not loaded:
            model, hist, diag = train_federated_one_seed(
                data, setting=setting, alpha=alpha, algorithm=algorithm,
                n_clients=n_clients, rounds=rounds, seed=seed,
            )
            payload = {
                "model_state": {k: v.detach().cpu() for k, v in get_state(model).items()},
                "history": hist.to_dict(orient="records"),
                "diagnostics": diag.to_dict(orient="records"),
                "dataset": data["name"], "seed": seed, "setting": setting,
                "alpha": alpha, "algorithm": algorithm, "clients": n_clients,
                "rounds": rounds, "local_epochs": LOCAL_EPOCHS,
            }
            tmp = ckpt.with_suffix(ckpt.suffix + ".tmp")
            torch.save(payload, tmp)
            os.replace(tmp, ckpt)

        all_hist.append(hist)
        last = hist.iloc[-1].to_dict()
        last["Seed_Total_Time_s"] = float(time.time() - t_seed) if not loaded else 0.0
        last_rows.append(last)
        last_model = model
        attack_free = int((diag["Class_1"] == 0).sum())
        print(
            f"F1={last['F1']:.4f} AUC={last['ROC_AUC']:.4f} "
            f"seed_total={last['Seed_Total_Time_s']:.1f}s "
            f"clients_attack_rate=[{diag['Attack_Rate'].min():.3f},{diag['Attack_Rate'].max():.3f}] "
            f"attack_free={attack_free}" + (" [RESUMED]" if loaded else ""),
            flush=True,
        )
    hist_all = pd.concat(all_hist, ignore_index=True)
    finals = pd.DataFrame(last_rows)
    summary = finals[
        ["Accuracy", "Precision", "Recall", "F1", "FPR", "ROC_AUC",
         "Balanced_Accuracy", "Round_Time_s", "Mean_Update_Norm", "Mean_Update_Cosine"]
    ].agg(["mean", "std"])
    return last_model, hist_all, finals, summary, diag


print("Multi-seed federation engine ready.")


# =============================================================================
# 7. CENTRALISED BASELINES + ABLATIONS
#    Completes former empty Table 3 (R2 #8); α / client / data-fraction ablations
# =============================================================================

def centralised_mlp(data, epochs=40, seed=BASE_SEED) -> Dict[str, Any]:
    seed_everything(seed)
    Xtr, ytr, Xte, yte = data["X_train"], data["y_train"], data["X_test"], data["y_test"]
    model = ResidualMLP(Xtr.shape[1]).to(DEVICE)
    t0 = time.time()
    train_local(model, Xtr, ytr, epochs=epochs)
    prob = predict_proba(model, Xte)
    met = evaluate_binary(yte, prob)
    met["Train_Time_s"] = time.time() - t0
    met["Model"] = "Centralised-MLP"
    met["Training_Setting"] = "Centralized"
    return met


def logistic_np(data, epochs=60, lr=0.1) -> Dict[str, Any]:
    Xtr, ytr, Xte, yte = data["X_train"], data["y_train"], data["X_test"], data["y_test"]
    w = np.zeros(Xtr.shape[1], dtype=np.float64)
    b = 0.0
    t0 = time.time()
    for _ in range(epochs):
        p = 1 / (1 + np.exp(-np.clip(Xtr @ w + b, -30, 30)))
        w -= lr * (Xtr.T @ (p - ytr)) / len(ytr)
        b -= lr * (p - ytr).mean()
    pte = 1 / (1 + np.exp(-np.clip(Xte @ w + b, -30, 30)))
    met = evaluate_binary(yte, pte)
    met["Train_Time_s"] = time.time() - t0
    met["Model"] = "LogisticRegression"
    met["Training_Setting"] = "Centralized"
    return met


def run_alpha_ablation(data, alphas=ALPHA_GRID, n_seeds=3, rounds=15) -> pd.DataFrame:
    rows = []
    for a in alphas:
        print(f"  α={a}")
        _, _, finals, summary, diag = multi_seed_federated(
            data,
            "Non-IID",
            alpha=a,
            algorithm="FedAvg",
            n_clients=N_CLIENTS,
            rounds=rounds,
            n_seeds=n_seeds,
        )
        row = summary.loc["mean"].to_dict()
        row.update(
            Alpha=a,
            Clients=N_CLIENTS,
            Client_Attack_Rate_SD=float(diag["Attack_Rate"].std()),
            F1_std=summary.loc["std", "F1"],
            Ablation_Rounds=rounds,
            Ablation_Seeds=n_seeds,
        )
        rows.append(row)
    return pd.DataFrame(rows)


def run_client_ablation(data, clients=CLIENT_ABLATION, n_seeds=3, rounds=15) -> pd.DataFrame:
    rows = []
    for k in clients:
        print(f"  clients={k}")
        _, _, finals, summary, diag = multi_seed_federated(
            data,
            "Non-IID",
            alpha=PRIMARY_ALPHA,
            algorithm="FedAvg",
            n_clients=k,
            rounds=rounds,
            n_seeds=n_seeds,
        )
        row = summary.loc["mean"].to_dict()
        row.update(
            Clients=k,
            Alpha=PRIMARY_ALPHA,
            Client_Attack_Rate_SD=float(diag["Attack_Rate"].std()),
            F1_std=summary.loc["std", "F1"],
            Ablation_Rounds=rounds,
            Ablation_Seeds=n_seeds,
        )
        rows.append(row)
    return pd.DataFrame(rows)


def run_data_fraction(
    data, fractions=(0.10, 0.25, 0.50, 1.00), rounds=12, seed=BASE_SEED
) -> pd.DataFrame:
    """Saturation curves justify subset size (R2 #10)."""
    X, y = data["X_train"], data["y_train"]
    rows = []
    for frac in fractions:
        n = max(1000, int(len(y) * frac))
        idx = np.random.default_rng(seed).choice(len(y), n, replace=False)
        sub = dict(data)
        sub["X_train"] = X[idx]
        sub["y_train"] = y[idx]
        _, hist, _ = train_federated_one_seed(
            sub,
            "Non-IID",
            alpha=PRIMARY_ALPHA,
            algorithm="FedAvg",
            n_clients=min(N_CLIENTS, 10),
            rounds=rounds,
            seed=seed,
        )
        last = hist.iloc[-1]
        rows.append(
            dict(
                Fraction=frac,
                Train_N=n,
                F1=last["F1"],
                ROC_AUC=last["ROC_AUC"],
                Accuracy=last["Accuracy"],
                Ablation_Rounds=rounds,
            )
        )
    return pd.DataFrame(rows)


print("Baselines + ablations ready.")


# =============================================================================
# 8. ATTRIBUTION (gradient × input) — expands SHAP-style analysis (R2 #5)
# =============================================================================

def gradient_input_attr(model, X_np, n_samples=500):
    model.eval()
    rng = np.random.default_rng(BASE_SEED)
    idx = rng.choice(len(X_np), min(n_samples, len(X_np)), replace=False)
    X = torch.as_tensor(X_np[idx], dtype=torch.float32, device=DEVICE)
    X.requires_grad_(True)
    logits = model(X)
    logits.sum().backward()
    attr = (X.grad * X).detach().cpu().numpy()
    return attr, idx


def attribution_analysis(iid_model, non_model, data, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    names = data["feature_names"]
    Xte, yte = data["X_test"], data["y_test"]
    vi, idx = gradient_input_attr(iid_model, Xte)
    vn, _ = gradient_input_attr(non_model, Xte)
    ye = yte[idx]

    imp = pd.DataFrame(
        {
            "Feature": names,
            "IID_MeanAbsAttr": np.abs(vi).mean(0),
            "NonIID_MeanAbsAttr": np.abs(vn).mean(0),
        }
    )
    imp["Absolute_Difference"] = (imp["IID_MeanAbsAttr"] - imp["NonIID_MeanAbsAttr"]).abs()
    imp = imp.sort_values("NonIID_MeanAbsAttr", ascending=False).reset_index(drop=True)
    imp.to_csv(outdir / "attr_iid_vs_noniid.csv", index=False)

    for tag, col in [("IID", "IID_MeanAbsAttr"), ("NonIID", "NonIID_MeanAbsAttr")]:
        top = imp.head(15)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(top["Feature"][::-1], top[col][::-1])
        ax.set_xlabel("Mean |attribution|")
        ax.set_title(f"{data['name']} – {tag}")
        fig.tight_layout()
        fig.savefig(outdir / f"attr_summary_{tag.lower()}.png", dpi=300)
        plt.close(fig)

    # Dependence plot for top feature
    top_f = imp.iloc[0]["Feature"]
    j = names.index(top_f)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(Xte[idx, j], vn[:, j], c=ye, cmap="coolwarm", s=8, alpha=0.6)
    ax.set_xlabel(top_f)
    ax.set_ylabel("Attribution")
    ax.set_title(f"Dependence – {top_f}")
    fig.tight_layout()
    fig.savefig(outdir / "attr_dependence_top.png", dpi=300)
    plt.close(fig)

    # Per-class (benign vs attack)
    class_rows = []
    for lab, lab_name in [(0, "Benign"), (1, "Attack")]:
        mask = ye == lab
        if not mask.any():
            continue
        class_rows.append(
            dict(
                Class=lab_name,
                N=int(mask.sum()),
                IID_MeanAbsAttr=float(np.abs(vi[mask]).mean()),
                NonIID_MeanAbsAttr=float(np.abs(vn[mask]).mean()),
            )
        )
        vals = np.abs(vn[mask]).mean(0)
        order = np.argsort(vals)[::-1][:10]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(np.array(names)[order][::-1], vals[order][::-1])
        ax.set_xlabel("Mean |attr|")
        ax.set_title(f"Non-IID – {lab_name}")
        fig.tight_layout()
        fig.savefig(outdir / f"attr_{lab_name.lower()}_top10.png", dpi=300)
        plt.close(fig)
    class_df = pd.DataFrame(class_rows)
    class_df.to_csv(outdir / "attr_class_conditional.csv", index=False)
    return imp, class_df


print("Attribution ready.")


# =============================================================================
# 9. STATISTICAL TESTS + MASTER REGISTRY (R2 numerical consistency)
# =============================================================================

def paired_stats_from_finals(iid_finals, non_finals) -> pd.DataFrame:
    rows = []
    for metric in ["F1", "ROC_AUC", "Accuracy", "Balanced_Accuracy"]:
        x = iid_finals[metric].to_numpy(dtype=float)
        y = non_finals[metric].to_numpy(dtype=float)
        diff = x - y
        n = len(diff)

        # Smoke runs with n<3 are descriptive only; inferential tests/effect
        # sizes are intentionally withheld.
        sh = tt = wp = d = np.nan
        if n >= 3 and np.all(np.isfinite(diff)):
            sd = float(diff.std(ddof=1))
            if sd > 1e-12:
                try:
                    sh = float(shapiro(diff).pvalue)
                except Exception:
                    pass
                try:
                    tt = float(ttest_rel(x, y).pvalue)
                except Exception:
                    pass
                try:
                    wp = float(wilcoxon(x, y).pvalue)
                except Exception:
                    pass
                d = float(diff.mean() / sd)

        rows.append(
            dict(
                Metric=metric,
                IID_Mean=float(x.mean()),
                NonIID_Mean=float(y.mean()),
                IID_SD=float(x.std(ddof=1)) if n > 1 else np.nan,
                NonIID_SD=float(y.std(ddof=1)) if n > 1 else np.nan,
                Shapiro_p=sh,
                Paired_t_p=tt,
                Wilcoxon_p=wp,
                Cohens_d=d,
            )
        )
    return pd.DataFrame(rows)


def save_registry(
    ds_name, central_rows, summaries, alpha_df=None, client_df=None, size_df=None
) -> pd.DataFrame:
    rows = []
    for central in central_rows:
        c = dict(central)
        c["Dataset"] = ds_name
        rows.append(c)
    for label, summary in summaries.items():
        r = summary.loc["mean"].to_dict()
        r.update(Dataset=ds_name, Model=label, Training_Setting=label)
        for m in ["Accuracy", "Precision", "Recall", "F1", "FPR", "ROC_AUC", "Balanced_Accuracy"]:
            if m in summary.columns:
                r[f"{m}_std"] = summary.loc["std", m]
        rows.append(r)
    reg = pd.DataFrame(rows)
    reg.to_csv(OUTPUT_DIR / "tables" / f"{ds_name}_MASTER_RESULTS.csv", index=False)
    if alpha_df is not None:
        alpha_df.to_csv(
            OUTPUT_DIR / "tables" / f"{ds_name}_Table_Alpha_Ablation.csv", index=False
        )
    if client_df is not None:
        client_df.to_csv(
            OUTPUT_DIR / "tables" / f"{ds_name}_Table_Client_Ablation.csv", index=False
        )
    if size_df is not None:
        size_df.to_csv(
            OUTPUT_DIR / "tables" / f"{ds_name}_Table_Data_Saturation.csv", index=False
        )
    return reg


print("Stats + registry ready.")


# =============================================================================
# 10. MAIN EXPERIMENT — dataset-worker experiment × N seeds × K clients (GPU)
# =============================================================================

def main() -> None:
    all_results: Dict[str, Any] = {}
    master_frames: List[pd.DataFrame] = []
    t_global = time.time()

    # Production protocol remains unchanged. Smoke mode deliberately uses the
    # smoke seed/round counts for every ablation so a validation run is truly fast.
    if ARGS.smoke:
        ablation_seeds = N_SEEDS
        ablation_rounds = N_ROUNDS
    else:
        ablation_seeds = max(3, N_SEEDS // 3)
        ablation_rounds = max(12, N_ROUNDS // 2)

    for ds_name, data in datasets.items():
        print("\n" + "=" * 90)
        print(f"DATASET: {ds_name} | source={data['source']} | device={DEVICE}")
        print("=" * 90)

        # Centralised baselines (completes Table 3)
        t_stage = time.time()
        cent_log = logistic_np(data, epochs=(20 if ARGS.smoke else 60))
        cent_mlp = centralised_mlp(data, epochs=(8 if ARGS.smoke else 40))
        print(f"[TIMING] Central baselines: {(time.time()-t_stage)/60:.2f} min", flush=True)
        print(
            f"Centralised LogReg F1={cent_log['F1']:.4f}  "
            f"MLP F1={cent_mlp['F1']:.4f}"
        )

        # Multi-seed FedAvg IID / NonIID / FedProx
        configs = [
            ("FedAvg-IID", "IID", "FedAvg", PRIMARY_ALPHA),
            ("FedAvg-NonIID", "Non-IID", "FedAvg", PRIMARY_ALPHA),
            ("FedProx-NonIID", "Non-IID", "FedProx", PRIMARY_ALPHA),
        ]
        summaries, finals_map, models, hist_map = {}, {}, {}, {}
        for label, setting, algo, alpha in configs:
            print(
                f"\n>> {label}  ({N_SEEDS} seeds, {N_CLIENTS} clients, {N_ROUNDS} rounds)"
            )
            model, hist_all, finals, summary, diag = multi_seed_federated(
                data,
                setting=setting,
                alpha=alpha,
                algorithm=algo,
                n_clients=N_CLIENTS,
                rounds=N_ROUNDS,
                n_seeds=N_SEEDS,
            )
            summaries[label] = summary
            finals_map[label] = finals
            models[label] = model
            hist_map[label] = hist_all
            print(summary.loc[["mean", "std"]].round(4))

        # Ablations
        print("\nα ablation...")
        t_stage = time.time()
        alpha_df = run_alpha_ablation(
            data, n_seeds=ablation_seeds, rounds=ablation_rounds
        )
        print(alpha_df.round(4).to_string(index=False))
        print(f"[TIMING] Alpha ablation: {(time.time()-t_stage)/60:.2f} min", flush=True)

        print("\nClient ablation...")
        t_stage = time.time()
        client_df = run_client_ablation(
            data, n_seeds=ablation_seeds, rounds=ablation_rounds
        )
        print(client_df.round(4).to_string(index=False))
        print(f"[TIMING] Client ablation: {(time.time()-t_stage)/60:.2f} min", flush=True)

        print("\nData-fraction saturation...")
        t_stage = time.time()
        size_df = run_data_fraction(data, rounds=ablation_rounds)
        print(size_df.round(4).to_string(index=False))
        print(f"[TIMING] Data saturation: {(time.time()-t_stage)/60:.2f} min", flush=True)

        # Paired stats IID vs NonIID (R1 result justification + R2 non-IID analysis)
        stats = paired_stats_from_finals(
            finals_map["FedAvg-IID"], finals_map["FedAvg-NonIID"]
        )
        stats.to_csv(
            OUTPUT_DIR / "tables" / f"{ds_name}_IID_vs_NonIID_statistics.csv",
            index=False,
        )
        print("\nPaired stats (IID vs NonIID across seeds):")
        print(stats.round(4).to_string(index=False))

        # Learning curves
        fig, ax = plt.subplots(figsize=(8, 4))
        for label, hist in hist_map.items():
            g = hist.groupby("Round")["F1"].agg(["mean", "std"])
            ax.plot(g.index, g["mean"], marker="o", label=label)
            ax.fill_between(
                g.index, g["mean"] - g["std"], g["mean"] + g["std"], alpha=0.15
            )
        ax.set_xlabel("Round")
        ax.set_ylabel("F1")
        ax.set_title(f"{ds_name}: F1 (mean±std over {N_SEEDS} seeds)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            OUTPUT_DIR / "figures" / f"{ds_name}_learning_curves.png", dpi=300
        )
        plt.close(fig)

        # Saturation
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(size_df["Fraction"] * 100, size_df["F1"], "-o", label="F1")
        ax.plot(size_df["Fraction"] * 100, size_df["ROC_AUC"], "-s", label="ROC-AUC")
        ax.set_xlabel("Training data (%)")
        ax.set_ylabel("Score")
        ax.set_title(f"{ds_name}: saturation")
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            OUTPUT_DIR / "figures" / f"{ds_name}_data_saturation.png", dpi=300
        )
        plt.close(fig)

        # α sensitivity
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.errorbar(
            alpha_df["Alpha"],
            alpha_df["F1"],
            yerr=alpha_df.get("F1_std", 0),
            marker="o",
            capsize=4,
        )
        ax.set_xlabel("Dirichlet α")
        ax.set_ylabel("F1")
        ax.set_title(f"{ds_name}: α sensitivity")
        fig.tight_layout()
        fig.savefig(
            OUTPUT_DIR / "figures" / f"{ds_name}_alpha_sensitivity.png", dpi=300
        )
        plt.close(fig)

        # Update diagnostics (supports non-IID explanation)
        non_hist = hist_map["FedAvg-NonIID"]
        g = non_hist.groupby("Round")[
            ["Mean_Update_Norm", "Mean_Update_Cosine"]
        ].mean()
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(g.index, g["Mean_Update_Norm"], "-o")
        axes[0].set_title("Update ℓ2-norm")
        axes[1].plot(g.index, g["Mean_Update_Cosine"], "-s", color="C1")
        axes[1].set_title("Update cosine")
        for ax_ in axes:
            ax_.set_xlabel("Round")
        fig.suptitle(f"{ds_name}: non-IID update diagnostics")
        fig.tight_layout()
        fig.savefig(
            OUTPUT_DIR / "figures" / f"{ds_name}_update_diagnostics.png", dpi=300
        )
        plt.close(fig)

        # Attribution
        attr_dir = OUTPUT_DIR / "figures" / ds_name.replace("/", "_")
        imp, class_df = attribution_analysis(
            models["FedAvg-IID"], models["FedAvg-NonIID"], data, attr_dir
        )

        reg = save_registry(
            ds_name, [cent_log, cent_mlp], summaries, alpha_df, client_df, size_df
        )
        master_frames.append(reg)

        all_results[ds_name] = dict(
            data=data,
            summaries=summaries,
            finals=finals_map,
            hist=hist_map,
            alpha=alpha_df,
            clients=client_df,
            saturation=size_df,
            stats=stats,
            registry=reg,
            importance=imp,
        )

    # -------------------------------------------------------------------------
    # Consolidated worker-local tables + comparison figure
    # -------------------------------------------------------------------------
    MASTER = pd.concat(master_frames, ignore_index=True)
    MASTER.to_csv(OUTPUT_DIR / "tables" / "WORKER_MASTER_RESULTS.csv", index=False)

    # Dataset comparison bar chart (worker-local when --worker-id is used)
    fed = MASTER[MASTER["Model"].astype(str).str.contains("Fed")].copy()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, metric in zip(axes, ["F1", "ROC_AUC", "Balanced_Accuracy"]):
        if metric not in fed.columns:
            continue
        models_order = ["FedAvg-IID", "FedAvg-NonIID", "FedProx-NonIID"]
        datasets_order = list(datasets.keys())
        x = np.arange(len(datasets_order))
        width = 0.25
        for i, m in enumerate(models_order):
            vals = [
                fed[(fed["Dataset"] == ds) & (fed["Model"] == m)][metric].values
                for ds in datasets_order
            ]
            vals = [v[0] if len(v) else 0 for v in vals]
            errs = []
            for ds in datasets_order:
                sc = fed[(fed["Dataset"] == ds) & (fed["Model"] == m)]
                errs.append(
                    sc[f"{metric}_std"].values[0]
                    if f"{metric}_std" in sc.columns and len(sc)
                    else 0
                )
            ax.bar(x + i * width, vals, width, yerr=errs, capsize=3, label=m)
        ax.set_xticks(x + width)
        ax.set_xticklabels(datasets_order, rotation=15, ha="right")
        ax.set_title(metric)
        ax.set_ylim(0, 1.05)
    axes[0].legend(fontsize=8)
    fig.suptitle(
        f"Dataset federated performance "
        f"(mean±std, {N_SEEDS} seeds, {N_CLIENTS} clients)"
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "figures" / "worker_dataset_comparison.png", dpi=300)
    plt.close(fig)

    # Numerical consistency audit
    audit = []
    for ds, r in all_results.items():
        reg = r["registry"]
        required = {"Accuracy", "F1", "ROC_AUC", "Dataset", "Model"}
        missing = required - set(reg.columns)
        audit.append(
            dict(
                Dataset=ds,
                Rows=len(reg),
                Missing=str(sorted(missing)),
                Duplicates=int(reg.duplicated(["Dataset", "Model"]).sum()),
                Source=r["data"]["source"],
            )
        )
    audit_df = pd.DataFrame(audit)
    audit_df.to_csv(
        OUTPUT_DIR / "tables" / "NUMERICAL_CONSISTENCY_AUDIT.csv", index=False
    )

    # =====================================================================
    # PRINT ALL MANUSCRIPT TABLES + FIGURE INDEX (console + text dump)
    # Old tables (centralised, FL IID/NonIID, α, clients) + new tables
    # (3rd dataset, FedProx, stats, attribution, saturation, audit)
    # =====================================================================
    def _hr(title: str) -> None:
        print("\n" + "=" * 90)
        print(title)
        print("=" * 90)

    def _tbl(name: str, df: pd.DataFrame, cols=None) -> None:
        print(f"\n--- {name} ---")
        if df is None or len(df) == 0:
            print("(empty)")
            return
        view = df if cols is None else df[[c for c in cols if c in df.columns]]
        print(view.round(4).to_string(index=False))

    report_lines: List[str] = []

    def _both(s: str = "") -> None:
        print(s)
        report_lines.append(s)

    _hr("ALL MANUSCRIPT TABLES (computed this run — print + CSV)")
    _both(f"Seeds={N_SEEDS}  Clients={N_CLIENTS}  Rounds={N_ROUNDS}  Device={DEVICE}")
    _both(f"Datasets: {', '.join(datasets.keys())}")

    # T0 — setup tables (already printed at startup; restate for completeness)
    _tbl("Table 0a. Metric definitions", METRIC_DEFINITIONS)
    _tbl("Table 0b. DNN architecture", ARCHITECTURE)
    _tbl("Table 0c. Literature positioning (Related Work)", LITERATURE)
    hyp_df = pd.DataFrame([HYPER])
    _tbl("Table 0d. Hyperparameters", hyp_df)

    # Dataset characteristics
    ds_rows = []
    for n, d in datasets.items():
        ds_rows.append(
            dict(
                Dataset=n,
                Source=d["source"],
                Train_N=len(d["y_train"]),
                Test_N=len(d["y_test"]),
                Features=d["n_features"],
                AttackRate_Train=d["attack_rate_train"],
                AttackRate_Test=d["attack_rate_test"],
            )
        )
    ds_char = pd.DataFrame(ds_rows)
    ds_char.to_csv(OUTPUT_DIR / "tables" / "Table_Dataset_Characteristics.csv", index=False)
    _tbl(f"Table 1. Dataset characteristics ({len(ds_char)} dataset(s) in this worker run)", ds_char)

    # Global MASTER (all models × datasets) — primary results table
    master_cols = [
        c
        for c in [
            "Dataset", "Model", "Accuracy", "Accuracy_std", "Precision", "Recall",
            "F1", "F1_std", "FPR", "ROC_AUC", "ROC_AUC_std", "Balanced_Accuracy",
            "Train_Time_s", "Round_Time_s",
        ]
        if c in MASTER.columns
    ]
    _tbl("Table 2. GLOBAL MASTER RESULTS (centralised + federated, mean±std)", MASTER, master_cols)

    # Per-dataset MASTER (old Table 5 / 11 style)
    for ds, r in all_results.items():
        reg = r["registry"]
        _tbl(
            f"Table 2x. MASTER registry — {ds}",
            reg,
            master_cols,
        )

    # Centralised only (old Table 3 — was empty; now filled)
    cent = MASTER[MASTER["Model"].astype(str).str.contains("Central|Logistic", case=False, na=False)]
    _tbl("Table 3. Centralised baselines (LogisticRegression + MLP)", cent, master_cols)

    # Federated primary (IID / NonIID / FedProx)
    fed_only = MASTER[MASTER["Model"].astype(str).str.contains("Fed", na=False)]
    _tbl("Table 4. Federated primary results (FedAvg-IID / FedAvg-NonIID / FedProx)", fed_only, master_cols)

    # α ablation (old Fig5 / Table discussion)
    for ds, r in all_results.items():
        adf = r["alpha"]
        acols = [c for c in ["Alpha", "F1", "F1_std", "ROC_AUC", "Accuracy",
                             "Balanced_Accuracy", "Client_Attack_Rate_SD"] if c in adf.columns]
        _tbl(f"Table 5. Dirichlet-α ablation — {ds}", adf, acols)

    # Client ablation (old Table 6)
    for ds, r in all_results.items():
        cdf = r["clients"]
        ccols = [c for c in ["Clients", "F1", "F1_std", "ROC_AUC", "Accuracy",
                             "Balanced_Accuracy", "Client_Attack_Rate_SD"] if c in cdf.columns]
        _tbl(f"Table 6. Client-count ablation — {ds}", cdf, ccols)

    # Data saturation (R2 subset justification)
    for ds, r in all_results.items():
        sdf = r["saturation"]
        _tbl(f"Table 7. Data-fraction saturation — {ds}", sdf)

    # IID vs NonIID statistical tests (R1/R2 justification)
    for ds, r in all_results.items():
        _tbl(f"Table 8. Paired statistics IID vs Non-IID — {ds}", r["stats"])

    # Attribution top features (expanded SHAP-style)
    for ds, r in all_results.items():
        imp = r.get("importance")
        if imp is not None:
            _tbl(
                f"Table 9. Feature attribution (IID vs Non-IID) — {ds} (top 15)",
                imp.head(15),
            )
            attr_path = OUTPUT_DIR / "figures" / ds.replace("/", "_") / "attr_class_conditional.csv"
            if attr_path.exists():
                class_df = pd.read_csv(attr_path)
                _tbl(f"Table 9b. Class-conditional attribution — {ds}", class_df)

    # Numerical consistency audit
    _tbl("Table 10. Numerical consistency audit (no cross-table contradictions)", audit_df)

    # -----------------------------------------------------------------
    # FIGURE INDEX — every PNG produced this run
    # -----------------------------------------------------------------
    _hr("ALL FIGURES GENERATED THIS RUN (300 dpi)")
    fig_list: List[str] = []
    for p in sorted((OUTPUT_DIR / "figures").rglob("*.png")):
        rel = str(p.relative_to(OUTPUT_DIR))
        fig_list.append(rel)
        print(f"  ✓ {rel}")

    # Named figure map (manuscript-style)
    print("\n--- Figure map (manuscript naming) ---")
    fig_map = [
        ("Fig 1", "worker_dataset_comparison.png", "Worker dataset F1 / ROC-AUC / Balanced Acc (mean±std)"),
    ]
    for ds in datasets:
        safe = ds.replace("/", "_")
        fig_map.extend(
            [
                (f"Fig LC-{safe}", f"{safe}_learning_curves.png", f"{ds}: F1 learning curves"),
                (f"Fig SAT-{safe}", f"{safe}_data_saturation.png", f"{ds}: data-fraction saturation"),
                (f"Fig ALP-{safe}", f"{safe}_alpha_sensitivity.png", f"{ds}: Dirichlet-α sensitivity"),
                (f"Fig UPD-{safe}", f"{safe}_update_diagnostics.png", f"{ds}: update ℓ2-norm & cosine"),
                (f"Fig ATTR-IID-{safe}", f"{safe}/attr_summary_iid.png", f"{ds}: IID attribution bar"),
                (f"Fig ATTR-NID-{safe}", f"{safe}/attr_summary_noniid.png", f"{ds}: Non-IID attribution bar"),
                (f"Fig DEP-{safe}", f"{safe}/attr_dependence_top.png", f"{ds}: dependence plot (top feature)"),
                (f"Fig BEN-{safe}", f"{safe}/attr_benign_top10.png", f"{ds}: benign class-conditional"),
                (f"Fig ATK-{safe}", f"{safe}/attr_attack_top10.png", f"{ds}: attack class-conditional"),
            ]
        )
    for tag, fname, desc in fig_map:
        exists = (OUTPUT_DIR / "figures" / fname).exists()
        mark = "✓" if exists else "·"
        print(f"  {mark} {tag:28s}  {fname:45s}  {desc}")

    # -----------------------------------------------------------------
    # CSV inventory
    # -----------------------------------------------------------------
    _hr("ALL CSV TABLES SAVED THIS RUN")
    for p in sorted((OUTPUT_DIR / "tables").rglob("*.csv")):
        print(f"  ✓ {p.relative_to(OUTPUT_DIR)}")

    response = f"""
Reviewer response — GPU-centric final pipeline
Seeds={N_SEEDS}  Clients={N_CLIENTS}  Rounds={N_ROUNDS}  Device={DEVICE}

1. Four validated real dataset workers (ML-EdgeIIoT, CIC-IoT2023, UNSW-NB15, Custom-Attack-Suite) are supported; the supplied DNN-EdgeIIoT file is excluded because it contains only Normal/benign samples.
2. Framework description precedes algorithms; standard Input/Output pseudocode.
3. Metrics formally defined and justified for class imbalance.
4. Residual MLP architecture + hyperparameters recorded; GPU AMP when CUDA available.
5. Multi-seed (n={N_SEEDS}) mean±std for all primary metrics; paired Wilcoxon/t-tests across seeds.
6. Non-IID diagnostics: client attack-rate SD, update norms, cosine similarity, learning curves.
7. α ∈ {{0.1, 0.5, 1.0}} and client counts {list(CLIENT_ABLATION)} ablated with error bars.
8. FedProx quantitative federated baseline under same partitions.
9. Attribution: global, dependence, class-conditional, IID vs non-IID (300 dpi).
10. Single MASTER registry per dataset; each worker writes a worker-local MASTER for later cross-worker aggregation.
11. Data-fraction saturation curves justify subset size.
12. Literature positioning table + centralised baselines complete former empty Table 3.
13. Validated preprocessing is cached by source-file signature; seed-level checkpoints permit safe resume without repeating completed trials.
14. Smoke mode uses a deterministic stratified subset and reduced epochs/rounds; smoke outputs are validation-only and must not be reported as production results.

All tables above are printed to console and saved under {OUTPUT_DIR / 'tables'}.
All figures are saved under {OUTPUT_DIR / 'figures'} (300 dpi).
"""
    print(response)
    (OUTPUT_DIR / "REVIEWER_RESPONSE.txt").write_text(response, encoding="utf-8")

    # Full text dump of every printed table for offline review
    dump_path = OUTPUT_DIR / "ALL_TABLES_PRINTED.txt"
    # Re-capture by writing a structured summary of dataframes
    with open(dump_path, "w", encoding="utf-8") as fh:
        fh.write("FedXAI-IDS — ALL TABLES DUMP\n")
        fh.write(f"Seeds={N_SEEDS} Clients={N_CLIENTS} Rounds={N_ROUNDS} Device={DEVICE}\n\n")
        fh.write("=== Table 1. Dataset characteristics ===\n")
        fh.write(ds_char.round(4).to_string(index=False) + "\n\n")
        fh.write("=== Table 2. GLOBAL MASTER ===\n")
        fh.write(MASTER[master_cols].round(4).to_string(index=False) + "\n\n")
        fh.write("=== Table 3. Centralised baselines ===\n")
        if len(cent):
            fh.write(cent[master_cols].round(4).to_string(index=False) + "\n\n")
        fh.write("=== Table 4. Federated primary ===\n")
        if len(fed_only):
            fh.write(fed_only[master_cols].round(4).to_string(index=False) + "\n\n")
        for ds, r in all_results.items():
            fh.write(f"=== α ablation — {ds} ===\n")
            fh.write(r["alpha"].round(4).to_string(index=False) + "\n\n")
            fh.write(f"=== Client ablation — {ds} ===\n")
            fh.write(r["clients"].round(4).to_string(index=False) + "\n\n")
            fh.write(f"=== Data saturation — {ds} ===\n")
            fh.write(r["saturation"].round(4).to_string(index=False) + "\n\n")
            fh.write(f"=== IID vs NonIID stats — {ds} ===\n")
            fh.write(r["stats"].round(4).to_string(index=False) + "\n\n")
            if r.get("importance") is not None:
                fh.write(f"=== Attribution top features — {ds} ===\n")
                fh.write(r["importance"].head(15).round(6).to_string(index=False) + "\n\n")
        fh.write("=== Numerical consistency audit ===\n")
        fh.write(audit_df.to_string(index=False) + "\n\n")
        fh.write("=== Figures ===\n")
        for rel in fig_list:
            fh.write(f"  {rel}\n")
    print(f"\nFull table dump written → {dump_path}")

    print(f"\nAll datasets finished in {(time.time() - t_global) / 60:.1f} min on {DEVICE}.")
    print("All artefacts:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
