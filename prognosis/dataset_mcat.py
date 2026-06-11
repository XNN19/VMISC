import hashlib
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def custom_hash(value, max_value=(2**63 - 1)):
    hash_obj = hashlib.sha256(str(value).encode())
    return int(hash_obj.hexdigest(), 16) % max_value


class MCAT_Survival_Dataset(Dataset):
    """Two-modality survival dataset used by the MCAT training scripts."""

    def __init__(
        self,
        df,
        data_dir_path,
        data_dir_omic,
        label_col="survival_months",
        n_bins=4,
        eps=1e-6,
        omic_suffix="_feats.pt",
        he_suffix=".pt",
    ):
        self.slide_data = df.copy()
        self.data_dir_path = data_dir_path
        self.data_dir_omic = data_dir_omic
        self.label_col = label_col
        self.n_bins = n_bins
        self.eps = eps
        self.omic_suffix = omic_suffix
        self.he_suffix = he_suffix

        rename_dict = {
            "patient_id": "case_id",
            "filename": "slide_id",
            "sex": "is_female",
            "time": "survival_months",
            "status": "status",
        }
        rename_dict = {k: v for k, v in rename_dict.items() if k in self.slide_data.columns}
        self.slide_data.rename(columns=rename_dict, inplace=True)

        if "slide_id" not in self.slide_data.columns and "case_id" in self.slide_data.columns:
            self.slide_data["slide_id"] = self.slide_data["case_id"]

        if "slide_id" in self.slide_data.columns:
            self.slide_data["slide_id"] = self.slide_data["slide_id"].apply(
                lambda x: x.replace(".svs", "") if isinstance(x, str) else x
            )

        if "status" in self.slide_data.columns and "censorship" not in self.slide_data.columns:
            self.slide_data["censorship"] = 1 - self.slide_data["status"]

        self._ensure_survival_labels()
        self.slide_cls_ids = [np.where(self.slide_data["label"] == i)[0] for i in range(n_bins)]

    def _ensure_survival_labels(self):
        if self.label_col not in self.slide_data.columns or "censorship" not in self.slide_data.columns:
            self.slide_data["label"] = 0
            self.slide_data["disc_label"] = 0
            return

        patients_df = self.slide_data.drop_duplicates(["case_id"]).copy()
        uncensored_df = patients_df[patients_df["censorship"] < 1]

        try:
            _, q_bins = pd.qcut(uncensored_df[self.label_col], q=self.n_bins, retbins=True, labels=False)
            q_bins[-1] = self.slide_data[self.label_col].max() + self.eps
            q_bins[0] = self.slide_data[self.label_col].min() - self.eps
            disc_labels = pd.cut(
                patients_df[self.label_col],
                bins=q_bins,
                labels=False,
                right=False,
                include_lowest=True,
            )
            patients_df.insert(2, "label", disc_labels.values.astype(int))
            self.slide_data = self.slide_data.merge(patients_df[["case_id", "label"]], on="case_id", how="left")
            self.slide_data["disc_label"] = self.slide_data["label"]
        except Exception as exc:
            print(f"Discretization failed: {exc}. Using 0 as label.")
            self.slide_data["label"] = 0
            self.slide_data["disc_label"] = 0

    def getlabel(self, ids):
        return self.slide_data["label"][ids]

    def __len__(self):
        return len(self.slide_data)

    def _normalize_patient_id(self, patient_id):
        if isinstance(patient_id, np.generic):
            patient_id = patient_id.item()
        if isinstance(patient_id, str):
            return custom_hash(patient_id)
        if isinstance(patient_id, float) and np.isnan(patient_id):
            return custom_hash("unknown")
        if isinstance(patient_id, (np.floating, np.integer)):
            return int(patient_id)
        if isinstance(patient_id, int):
            return patient_id
        return custom_hash(patient_id)

    def _load_tensor(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing feature tensor: {path}")
        try:
            feat = torch.load(path, weights_only=False)
        except TypeError:
            feat = torch.load(path)
        if isinstance(feat, np.ndarray):
            feat = torch.from_numpy(feat)
        return feat

    def __getitem__(self, idx):
        row = self.slide_data.iloc[idx]
        case_id = self._normalize_patient_id(row["case_id"])
        slide_id = row["slide_id"]

        w_feat = self._load_tensor(os.path.join(self.data_dir_path, f"{slide_id}{self.he_suffix}"))
        o_feat = self._load_tensor(os.path.join(self.data_dir_omic, f"{slide_id}{self.omic_suffix}"))

        label = int(row.get("disc_label", 0))
        event_time = float(row.get(self.label_col, 0))
        censorship = float(row.get("censorship", 0))
        return [w_feat, o_feat, label, event_time, censorship, case_id]
