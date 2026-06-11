import argparse
import atexit
import glob
import os
from pathlib import Path


_WORKER_WSI = None
_RUNTIME_LOADED = False


def _load_runtime():
    """Import heavy inference dependencies only when inference is executed."""
    global _RUNTIME_LOADED
    if _RUNTIME_LOADED:
        return

    global torch, yaml, pd, openslide, h5py, np
    global DataLoader, transforms, pl, BasePredictionWriter
    global PlexModel, Virchow2Lightning, trnsfrms_val

    from models import PlexModel
    from models.virchow2lightning import Virchow2Lightning

    import h5py
    import numpy as np
    import openslide
    import pandas as pd
    import pytorch_lightning as pl
    import torch
    import yaml
    from pytorch_lightning.callbacks import BasePredictionWriter
    from torch.utils.data import DataLoader
    from torchvision import transforms

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    trnsfrms_val = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    _RUNTIME_LOADED = True


def _worker_cleanup():
    global _WORKER_WSI
    try:
        if _WORKER_WSI is not None and hasattr(_WORKER_WSI, "close"):
            _WORKER_WSI.close()
    finally:
        _WORKER_WSI = None


atexit.register(_worker_cleanup)


def worker_init_fn(_):
    global _WORKER_WSI
    _WORKER_WSI = None


def find_slide_path(he_dir, img_id, extension=".svs"):
    candidates = (
        glob.glob(os.path.join(he_dir, f"{img_id}{extension}"))
        + glob.glob(os.path.join(he_dir, "TCGA-COAD", f"{img_id}{extension}"))
        + glob.glob(os.path.join(he_dir, "TCGA-READ", f"{img_id}{extension}"))
    )
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected one slide for {img_id}, found {len(candidates)}: {candidates}")
    return candidates[0]


def get_patch_size(he_dir, img_id, extension=".svs"):
    _load_runtime()
    fpath = find_slide_path(he_dir, img_id, extension)
    he_wsi = openslide.open_slide(fpath)
    try:
        mpp = he_wsi.properties[openslide.PROPERTY_NAME_MPP_X]
    finally:
        if hasattr(he_wsi, "close"):
            he_wsi.close()
    physical = 512 * 0.25
    return int(physical / float(mpp))


class InferenceDataset:
    def __init__(self, he_dir, img_id, coords_xy, patch_size, extension=".svs"):
        _load_runtime()
        self.he_path = find_slide_path(he_dir, img_id, extension)
        self.img_id = img_id
        self.coords = coords_xy.astype(np.int64, copy=False)
        self.patch_size = int(patch_size)
        self.transform = trnsfrms_val

    def _get_wsi(self):
        global _WORKER_WSI
        if _WORKER_WSI is None:
            _WORKER_WSI = openslide.open_slide(self.he_path)
        return _WORKER_WSI

    def __len__(self):
        return self.coords.shape[0]

    def __getitem__(self, idx):
        wsi = self._get_wsi()
        x, y = self.coords[idx]
        patch = wsi.read_region((int(x), int(y)), 0, (self.patch_size, self.patch_size)).convert("RGB")
        he_image = self.transform(patch).float()
        highplex_placeholder = ""
        return he_image, highplex_placeholder, self.img_id, np.array([x, y], dtype=np.int64)


def make_dual_predictor():
    _load_runtime()

    class DualPredictor(pl.LightningModule):
        def __init__(self, model_19plex, model_static):
            super().__init__()
            self.model_19plex = model_19plex
            self.model_static = model_static

        def predict_step(self, batch, batch_idx, dataloader_idx=0):
            he, plex, img_id, coord_xy = batch
            with torch.inference_mode():
                tmp_static = self.model_static(he)
                out_19plex = self.model_19plex((he, plex))

            class_token = tmp_static[:, 0]
            patch_tokens = tmp_static[:, 5:]
            out_static = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)

            if isinstance(out_19plex, dict) and "logit" in out_19plex:
                out_19plex_logit = out_19plex["logit"]
                out_19plex_feat = out_19plex["embedding"]
            else:
                out_19plex_logit = out_19plex
                out_19plex_feat = torch.empty((out_19plex_logit.shape[0], 0), device=out_19plex_logit.device)

            return {
                "out_19plex": out_19plex_logit,
                "out_19plex_feat": out_19plex_feat,
                "out_static": out_static,
                "img_id": img_id,
                "coord_xy": coord_xy,
            }

    return DualPredictor


def make_inference_writer():
    _load_runtime()

    class InferenceWriter(BasePredictionWriter):
        def __init__(self, output_dir, write_interval):
            super().__init__(write_interval)
            self.output_dir_19plex = os.path.join(output_dir, "19plex")
            self.output_dir_19plex_feat = os.path.join(output_dir, "19plex_feat")
            self.output_dir_static = os.path.join(output_dir, "static")
            os.makedirs(self.output_dir_19plex, exist_ok=True)
            os.makedirs(self.output_dir_19plex_feat, exist_ok=True)
            os.makedirs(self.output_dir_static, exist_ok=True)
            self.reset()

        def reset(self):
            self._preds_19plex = []
            self._preds_19plex_feat = []
            self._preds_static = []
            self._img_ids = []
            self._coords = []

        def write_on_batch_end(self, trainer, pl_module, predictions, batch_indices, batch, batch_idx, dataloader_idx=0):
            pred_19plex = predictions["out_19plex"].detach().cpu()
            pred_19plex_feat = predictions["out_19plex_feat"].detach().cpu()
            pred_static = predictions["out_static"].detach().cpu()

            pred_19plex = pred_19plex.unsqueeze(0) if pred_19plex.dim() == 1 else pred_19plex
            pred_19plex_feat = pred_19plex_feat.unsqueeze(0) if pred_19plex_feat.dim() == 1 else pred_19plex_feat
            pred_static = pred_static.unsqueeze(0) if pred_static.dim() == 1 else pred_static

            coord = predictions["coord_xy"]
            self._preds_19plex.append(pred_19plex)
            self._preds_19plex_feat.append(pred_19plex_feat)
            self._preds_static.append(pred_static)
            self._img_ids.extend(list(predictions["img_id"]))
            self._coords.append(coord.detach().cpu().numpy() if hasattr(coord, "detach") else np.asarray(coord))

        def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
            img_ids = sorted(set(self._img_ids))
            if len(img_ids) != 1:
                raise ValueError(f"Expected predictions from one slide per epoch, got {img_ids}")
            img_id = img_ids[0]

            preds_19plex = torch.cat(self._preds_19plex, dim=0)
            preds_19plex_feat = torch.cat(self._preds_19plex_feat, dim=0)
            preds_static = torch.cat(self._preds_static, dim=0)

            coords = np.concatenate(self._coords, axis=0)
            coord_strs = [f"{coord[0]}_{coord[1]}" for coord in coords]
            df = pd.DataFrame({"img_id": [img_id] * len(coord_strs), "coord": coord_strs})

            torch.save(preds_19plex, os.path.join(self.output_dir_19plex, f"{img_id}_preds.pt"))
            df.to_csv(os.path.join(self.output_dir_19plex, f"{img_id}_indices.csv"), index=False)

            torch.save(preds_19plex_feat, os.path.join(self.output_dir_19plex_feat, f"{img_id}_preds.pt"))
            df.to_csv(os.path.join(self.output_dir_19plex_feat, f"{img_id}_indices.csv"), index=False)

            torch.save(preds_static, os.path.join(self.output_dir_static, f"{img_id}_preds.pt"))
            df.to_csv(os.path.join(self.output_dir_static, f"{img_id}_indices.csv"), index=False)
            self.reset()

    return InferenceWriter


def load_config(config_path):
    _load_runtime()
    with open(config_path, "r", encoding="utf-8") as handle:
        return dict(yaml.load(handle, Loader=yaml.Loader))


def load_coordinates(patch_dir, slide_id):
    _load_runtime()
    h5_path = os.path.join(patch_dir, f"{slide_id}.h5")
    with h5py.File(h5_path, "r") as handle:
        keys = list(handle.keys())
        if not keys:
            raise ValueError(f"No datasets found in {h5_path}")
        coords = handle[keys[0]][:]
    df = pd.DataFrame(coords, columns=["coord_x", "coord_y"])
    df["img_id"] = slide_id
    return df


def discover_slides(patch_dir, he_dir, ext, selected_slides=None):
    if selected_slides:
        return selected_slides

    h5_slides = sorted(Path(patch_dir).glob("*.h5")) if patch_dir else []
    if h5_slides:
        return [path.stem for path in h5_slides]

    roots = [Path(he_dir), Path(he_dir) / "TCGA-COAD", Path(he_dir) / "TCGA-READ"]
    slide_ids = []
    for root in roots:
        if root.exists():
            slide_ids.extend(path.name[: -len(ext)] for path in root.glob(f"*{ext}"))
    return sorted(set(slide_ids))


def inference_func(
    model_19plex,
    model_static,
    checkpoint_dir,
    output_dir,
    he_dir,
    valid_df,
    devices,
    accelerator="auto",
    batch_size=1560,
    num_workers=8,
    precision="16-mixed",
    debug_mode=False,
    write_interval="batch_and_epoch",
    ext=".svs",
    skip_existing=True,
):
    _load_runtime()
    checkpoint_path = os.path.join(checkpoint_dir, "vmisc.pth")
    map_location = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        state_dict = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    except TypeError:
        state_dict = torch.load(checkpoint_path, map_location=map_location)
    model_19plex.load_state_dict(state_dict)
    model_19plex.eval()
    model_static.eval()

    if debug_mode:
        valid_df = valid_df.iloc[:100]

    DualPredictor = make_dual_predictor()
    InferenceWriter = make_inference_writer()
    dual = DualPredictor(model_19plex, model_static)
    dual_writer = InferenceWriter(output_dir, write_interval)

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        callbacks=[dual_writer],
        logger=False,
    )

    img_id = valid_df["img_id"].unique()[0]
    coords_xy = valid_df[["coord_x", "coord_y"]].to_numpy(dtype=np.int64, copy=False)
    patch_size = get_patch_size(he_dir, img_id, extension=ext)

    dataset = InferenceDataset(he_dir, img_id, coords_xy, patch_size, extension=ext)
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "pin_memory": accelerator == "gpu",
        "num_workers": num_workers,
        "worker_init_fn": worker_init_fn if num_workers > 0 else None,
        "persistent_workers": False,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 4

    dataloader = DataLoader(dataset, **loader_kwargs)
    trainer.predict(dual, dataloaders=dataloader, return_predictions=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Run VMISC marker WSI inference from CLAM h5 patch coordinates.")
    parser.add_argument("--config", default=os.path.join("configs", "examples", "marker.yaml"))
    parser.add_argument("--checkpoint-dir", required=True, help="Directory containing fold_0.pth.")
    parser.add_argument("--output-dir", required=True, help="Directory for 19plex, 19plex_feat, and static outputs.")
    parser.add_argument("--he-dir", required=True, help="Directory containing slides, or TCGA-COAD/TCGA-READ subfolders.")
    parser.add_argument("--patch-dir", required=True, help="Directory containing CLAM-style <slide_id>.h5 coordinate files.")
    parser.add_argument("--slides", nargs="*", default=None, help="Optional slide ids to process. Defaults to all h5 files.")
    parser.add_argument("--ext", default=".svs", help="Slide extension, for example .svs or .tif.")
    parser.add_argument("--devices", nargs="+", default="auto", help="PyTorch Lightning devices, for example 0 or 0 1.")
    parser.add_argument("--accelerator", default="auto", choices=["auto", "cpu", "gpu"])
    parser.add_argument("--precision", default="16-mixed")
    parser.add_argument("--batch-size", type=int, default=1560)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--debug", action="store_true", help="Process only the first 100 coordinates for each slide.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute slides even when output index files exist.")
    return parser.parse_args()


def normalize_devices(devices):
    if devices == "auto":
        return "auto"
    parsed = []
    for value in devices:
        try:
            parsed.append(int(value))
        except ValueError:
            return devices
    return parsed


def main():
    args = parse_args()
    _load_runtime()

    os.makedirs(args.output_dir, exist_ok=True)
    cfg = load_config(args.config)
    model = PlexModel(cfg, save_path=args.output_dir)
    model_static = Virchow2Lightning()

    slides = discover_slides(args.patch_dir, args.he_dir, args.ext, args.slides)
    if not slides:
        raise FileNotFoundError(f"No slides found in patch dir {args.patch_dir}")

    devices = normalize_devices(args.devices)
    for sidx, slide in enumerate(slides):
        print(f"{slide} {sidx + 1}/{len(slides)}")
        curr_df = load_coordinates(args.patch_dir, slide)
        inference_func(
            model,
            model_static,
            args.checkpoint_dir,
            args.output_dir,
            args.he_dir,
            curr_df,
            devices=devices,
            accelerator=args.accelerator,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            precision=args.precision,
            debug_mode=args.debug,
            ext=args.ext,
            skip_existing=not args.overwrite,
        )


if __name__ == "__main__":
    main()
