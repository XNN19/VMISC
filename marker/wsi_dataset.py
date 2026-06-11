import base64
import pickle
from copy import deepcopy
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms


IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)


def slide_id(path):
    return Path(path).stem.split("_")[0]


def index_he_embeddings(he_dir):
    index = {}
    for path in Path(he_dir).glob("*.pt"):
        img_id = slide_id(path)
        coord = Path(path).stem[len(img_id) + 1 :]
        index.setdefault(img_id, {})[coord] = path
    return index


def index_he_images(he_dir):
    index = {}
    for path in Path(he_dir).glob("*.tsv"):
        img_id = slide_id(path)
        with open(f"{path}.index", "rb") as handle:
            rows, offsets = pickle.load(handle)
        index[img_id] = {}
        with path.open("r", encoding="utf-8") as handle:
            for row in range(rows):
                handle.seek(offsets[row])
                image_b64, coord = handle.readline().rstrip("\n").split("\t")
                index[img_id][coord] = image_b64
    return index


def index_plex_targets(plex_dir):
    index = {}
    for slide_dir in Path(plex_dir).iterdir():
        if not slide_dir.is_dir():
            continue
        index[slide_dir.name] = {
            path.stem[len(slide_dir.name) + 1 :]: path for path in slide_dir.glob("*.npy")
        }
    return index


class PatchImageDataset(Dataset):
    def __init__(self, df, config):
        self.df = df.reset_index(drop=True)
        self.is_embed = bool(config["is_embed"])
        self.log_plex = bool(config["plex_log"])
        self.he = index_he_embeddings(config["he_dir"]) if self.is_embed else index_he_images(config["he_dir"])
        self.plex = index_plex_targets(config["plex_dir"])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        img_id, coord = item.img_id, item.coord

        if self.is_embed:
            he = torch.load(self.he[img_id][coord], map_location="cpu").float()
        else:
            image = Image.open(BytesIO(base64.urlsafe_b64decode(self.he[img_id][coord]))).convert("RGB")
            he = IMAGE_TRANSFORM(image).float()

        plex = np.load(self.plex[img_id][coord])
        plex = np.log1p(plex) if self.log_plex else plex
        return he, torch.as_tensor(plex, dtype=torch.float32)


class WSIDataModule(LightningDataModule):
    def __init__(self, config, split_k=0, dist=True):
        super().__init__()
        data_cfg = config["Data"]
        df = pd.read_csv(data_cfg["dataframe"])
        val_df = df[df["fold"] == split_k].reset_index(drop=True)
        train_df = df[df["fold"] != split_k].reset_index(drop=True)
        test_df = pd.read_csv(data_cfg["test_df"]) if data_cfg.get("test_df") else deepcopy(val_df)

        self.datasets = [PatchImageDataset(frame, data_cfg) for frame in (train_df, val_df, test_df)]
        self.batch_size = data_cfg["batch_size"]
        self.num_workers = data_cfg["num_workers"]
        self.dist = dist
        self.samplers = [None, None, None]

    def setup(self, stage=None):
        if self.dist:
            self.samplers = [DistributedSampler(dataset, shuffle=True) for dataset in self.datasets]

    def _loader(self, idx):
        return DataLoader(
            self.datasets[idx],
            batch_size=self.batch_size,
            sampler=self.samplers[idx],
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def train_dataloader(self):
        return self._loader(0)

    def val_dataloader(self):
        return self._loader(1)

    def test_dataloader(self):
        return self._loader(2)
