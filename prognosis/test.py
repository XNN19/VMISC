import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sksurv.metrics import concordance_index_censored
from torch.utils.data import DataLoader, SequentialSampler

from dataset_mcat import MCAT_Survival_Dataset
from models.model_coattn import MCAT_Surv
from utils import collate_MIL_survival_sig


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"YAML at {path} did not parse to a dictionary.")
    return cfg


def build_dataset(cfg, csv_path):
    df = pd.read_csv(csv_path)
    return MCAT_Survival_Dataset(
        df=df,
        data_dir_path=str(cfg["data_dir_path"]),
        data_dir_omic=str(cfg["data_dir_omic"]),
        label_col=str(cfg.get("label_col", "survival_months")),
        n_bins=int(cfg.get("n_classes", 4)),
        he_suffix=str(cfg.get("he_suffix", ".pt")),
        omic_suffix=str(cfg.get("omic_suffix", "_feats.pt")),
    )


def checkpoint_state_dict(path):
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        return state["model_state_dict"]
    return state


def evaluate(model, loader, n_classes, device):
    model.eval()
    rows = []

    with torch.no_grad():
        for batch_idx, (x_path, x_vmisc, label, event_time, censorship, case_id) in enumerate(loader):
            x_path = x_path.to(device)
            x_vmisc = x_vmisc.to(device)
            hazards, survival, _, _ = model(x_path=x_path, x_vmisc=x_vmisc)
            risk = float(-torch.sum(survival, dim=1).cpu().numpy()) if n_classes > 1 else float(hazards.cpu().numpy())

            rows.append(
                {
                    "case_id": str(case_id[0]) if isinstance(case_id, np.ndarray) else str(case_id),
                    "slide_id": loader.dataset.slide_data.iloc[batch_idx]["slide_id"],
                    "risk": risk,
                    "disc_label": int(label.item()),
                    "survival": float(event_time),
                    "censorship": float(censorship),
                }
            )

    results = pd.DataFrame(rows)
    c_index = concordance_index_censored(
        (1 - results["censorship"]).astype(bool),
        results["survival"],
        results["risk"],
        tied_tol=1e-8,
    )[0]
    grouped_mean = results.groupby("case_id").agg({"risk": "mean", "survival": "first", "censorship": "first"})
    grouped_max = results.groupby("case_id").agg({"risk": "max", "survival": "first", "censorship": "first"})
    c_index_mean = concordance_index_censored(
        (1 - grouped_mean["censorship"]).astype(bool),
        grouped_mean["survival"],
        grouped_mean["risk"],
        tied_tol=1e-8,
    )[0]
    c_index_max = concordance_index_censored(
        (1 - grouped_max["censorship"]).astype(bool),
        grouped_max["survival"],
        grouped_max["risk"],
        tied_tol=1e-8,
    )[0]
    return results, {"c_index": c_index, "c_index_mean": c_index_mean, "c_index_max": c_index_max}


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained VMISC prognosis checkpoint.")
    parser.add_argument("--config", required=True, help="YAML config used for prognosis data/model settings.")
    parser.add_argument("--checkpoint", required=True, help="Path to a trained MCAT checkpoint .pt file.")
    parser.add_argument("--csv-path", default=None, help="Evaluation CSV. Defaults to val_csv_path from config.")
    parser.add_argument("--output-dir", default="results/prognosis_test")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    eval_csv = args.csv_path or cfg.get("val_csv_path")
    if not eval_csv:
        raise ValueError("Provide --csv-path or set val_csv_path in the YAML config.")

    n_classes = int(cfg.get("n_classes", 4))
    fusion = cfg.get("fusion", "concat")
    model = MCAT_Surv(fusion=None if fusion == "None" else fusion, n_classes=n_classes)
    model.load_state_dict(checkpoint_state_dict(args.checkpoint))
    device = torch.device(args.device)
    model = model.to(device)

    dataset = build_dataset(cfg, eval_csv)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=SequentialSampler(dataset),
        collate_fn=collate_MIL_survival_sig,
    )
    results, metrics = evaluate(model, loader, n_classes, device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "predictions.csv", index=False)
    pd.DataFrame([metrics]).to_csv(output_dir / "metrics.csv", index=False)
    print(metrics)


if __name__ == "__main__":
    main()
