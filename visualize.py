import argparse
import csv
import pickle
import zipfile
import math
from pathlib import Path

import numpy as np


def require_tifffile():
    try:
        from tifffile import imwrite
    except ImportError as exc:
        raise ImportError(
            "This conversion requires tifffile. Install the project environment from environment.yml or requirements.txt."
        ) from exc
    return imwrite


DEFAULT_MARKERS = [
    "Hoechst",
    "AF1",
    "CD31",
    "CD45",
    "CD68",
    "Argo550",
    "CD4",
    "FOXP3",
    "CD8a",
    "CD45RO",
    "CD20",
    "PD-L1",
    "CD3e",
    "CD163",
    "E-cadherin",
    "PD-1",
    "Ki67",
    "Pan-CK",
    "SMA",
]


def to_u16_cyx(img_cyx):
    """Convert a channel-first float image in [0, 1] to uint16 CYX."""
    img_cyx = np.nan_to_num(img_cyx, nan=0.0, posinf=0.0, neginf=0.0)
    img_cyx = np.clip(img_cyx, 0.0, 1.0)
    return (img_cyx * 65535).round().astype(np.uint16)


def infer_patch_size(name, default_patch_size):
    if "AG-3898" in name:
        return 550
    if "-5M-" in name:
        return 506
    return default_patch_size


def parse_coord(coord):
    parts = str(coord).replace("x", "").replace("y", "").split("_")
    if len(parts) != 2:
        raise ValueError(f"Invalid coordinate string: {coord}")
    return int(parts[0]), int(parts[1])


def read_markers(markers_path):
    if markers_path is None:
        return DEFAULT_MARKERS
    with open(markers_path, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        rows = [row for row in reader if row]
    if not rows:
        raise ValueError(f"{markers_path} does not contain markers")
    if rows[0][0] == "marker_name":
        rows = rows[1:]
    return [row[0] for row in rows]


class _TorchStorageType:
    def __init__(self, dtype):
        self.dtype = np.dtype(dtype)


def _load_torch_zip_tensor(pt_path):
    storage_types = {
        ("torch", "HalfStorage"): _TorchStorageType(np.float16),
        ("torch", "FloatStorage"): _TorchStorageType(np.float32),
        ("torch", "DoubleStorage"): _TorchStorageType(np.float64),
        ("torch", "LongStorage"): _TorchStorageType(np.int64),
        ("torch", "IntStorage"): _TorchStorageType(np.int32),
        ("torch", "ShortStorage"): _TorchStorageType(np.int16),
        ("torch", "ByteStorage"): _TorchStorageType(np.uint8),
        ("torch", "CharStorage"): _TorchStorageType(np.int8),
        ("torch", "BoolStorage"): _TorchStorageType(bool),
    }

    with zipfile.ZipFile(pt_path) as archive:
        data_pkl = next(name for name in archive.namelist() if name.endswith("/data.pkl"))
        root = data_pkl[: -len("data.pkl")]

        def rebuild_tensor(storage, storage_offset, size, stride, requires_grad, backward_hooks):
            dtype = storage.dtype
            itemsize = dtype.itemsize
            offset = int(storage_offset) * itemsize
            shape = tuple(int(v) for v in size)
            byte_strides = tuple(int(v) * itemsize for v in stride)
            view = np.lib.stride_tricks.as_strided(storage.array[offset // itemsize :], shape=shape, strides=byte_strides)
            return np.array(view, copy=True)

        class TensorUnpickler(pickle.Unpickler):
            def persistent_load(self, pid):
                typename, storage_type, key, location, numel = pid
                if typename != "storage":
                    raise pickle.UnpicklingError(f"Unsupported persistent id: {pid}")
                dtype = storage_type.dtype
                raw = archive.read(f"{root}data/{key}")
                array = np.frombuffer(raw, dtype=dtype, count=int(numel))
                storage = type("LoadedStorage", (), {})()
                storage.dtype = dtype
                storage.array = array
                return storage

            def find_class(self, module, name):
                if (module, name) in storage_types:
                    return storage_types[(module, name)]
                if module == "torch._utils" and name == "_rebuild_tensor_v2":
                    return rebuild_tensor
                if module == "collections" and name == "OrderedDict":
                    from collections import OrderedDict

                    return OrderedDict
                raise pickle.UnpicklingError(f"Unsupported class in torch checkpoint: {module}.{name}")

        return TensorUnpickler(archive.open(data_pkl)).load()


def load_torch_tensor(pt_path):
    try:
        import torch
    except ImportError as exc:
        try:
            return np.asarray(_load_torch_zip_tensor(pt_path), dtype=np.float32)
        except Exception as fallback_exc:
            raise ImportError(
                "Reading this marker inference output requires torch. "
                "Install the project environment from environment.yml or requirements.txt."
            ) from fallback_exc

    try:
        tensor = torch.load(pt_path, map_location="cpu", weights_only=True)
    except TypeError:
        tensor = torch.load(pt_path, map_location="cpu")

    if isinstance(tensor, dict):
        if "logit" in tensor:
            tensor = tensor["logit"]
        elif "preds" in tensor:
            tensor = tensor["preds"]
        else:
            raise ValueError(f"{pt_path} contains a dict without a supported prediction key.")

    if hasattr(tensor, "detach"):
        tensor = tensor.detach().cpu().numpy()
    return np.asarray(tensor, dtype=np.float32)


def marker_grid_to_ometiff(image_name, coords_xy, values, output_dir=".", patch_size=507, markers=None, log1p_input=True):
    imwrite = require_tifffile()
    markers = markers or DEFAULT_MARKERS
    coords_xy = np.asarray(coords_xy, dtype=np.int64)
    values = np.asarray(values, dtype=np.float32)

    if values.ndim != 2:
        raise ValueError(f"{image_name}: expected a 2D prediction array, got shape {values.shape}")
    if values.shape[0] != len(coords_xy) and values.shape[1] == len(coords_xy):
        values = values.T
    if values.shape[0] != len(coords_xy):
        raise ValueError(f"{image_name}: {len(coords_xy)} coordinates but prediction shape is {values.shape}")
    if values.shape[1] != len(markers):
        raise ValueError(f"{image_name}: {values.shape[1]} channels but {len(markers)} markers were provided")

    xs = coords_xy[:, 0]
    ys = coords_xy[:, 1]
    h = math.ceil((int(ys.max()) + patch_size) / patch_size)
    w = math.ceil((int(xs.max()) + patch_size) / patch_size)

    print(f"{image_name}: grid={h}x{w}, channels={len(markers)}, patch_size={patch_size}")
    img = np.full((len(markers), h, w), np.nan, dtype=np.float32)

    if log1p_input:
        values = np.expm1(values)

    for row_idx, (x, y) in enumerate(coords_xy):
        gx = int(x // patch_size)
        gy = int(y // patch_size)
        if gy >= h or gx >= w:
            continue
        img[:, gy, gx] = values[row_idx]

    for channel_idx in range(len(markers)):
        channel = img[channel_idx]
        mask = ~np.isnan(channel)
        if mask.any():
            vmin, vmax = channel[mask].min(), channel[mask].max()
            if vmax > vmin:
                channel[mask] = (channel[mask] - vmin) / (vmax - vmin)
            img[channel_idx] = np.nan_to_num(channel, nan=0.0)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{image_name}_vmisc.ome.tiff"
    imwrite(
        output_path,
        to_u16_cyx(img),
        photometric="minisblack",
        metadata={
            "axes": "CYX",
            "Channel": [{"Name": marker} for marker in markers],
        },
    )
    print(f"Saved {output_path}")
    return output_path


def table_to_ometiff(tsv_path, output_dir=".", patch_size=507, markers=None, log1p_input=True):
    tsv_path = Path(tsv_path)
    markers = markers or DEFAULT_MARKERS

    with tsv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{tsv_path} does not contain marker rows")

    missing = {"X", "Y", *markers}.difference(rows[0].keys())
    if missing:
        raise ValueError(f"{tsv_path} is missing required columns: {sorted(missing)}")

    coords = np.array([[int(row["X"]), int(row["Y"])] for row in rows], dtype=np.int64)
    values = np.array([[float(row[marker]) for marker in markers] for row in rows], dtype=np.float32)

    return marker_grid_to_ometiff(
        tsv_path.stem,
        coords,
        values,
        output_dir=output_dir,
        patch_size=patch_size,
        markers=markers,
        log1p_input=log1p_input,
    )


def inference_output_to_ometiff(outputs_dir, output_dir=".", patch_size=507, markers=None, log1p_input=True):
    outputs_dir = Path(outputs_dir)
    marker_dir = outputs_dir / "19plex"
    if not marker_dir.is_dir():
        raise FileNotFoundError(f"Expected marker predictions in {marker_dir}")

    output_paths = []
    for preds_path in sorted(marker_dir.glob("*_preds.pt")):
        image_name = preds_path.name[: -len("_preds.pt")]
        indices_path = marker_dir / f"{image_name}_indices.csv"
        if not indices_path.exists():
            raise FileNotFoundError(f"Missing indices CSV for {preds_path}: {indices_path}")

        with indices_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"{indices_path} does not contain index rows")

        missing = {"img_id", "coord"}.difference(rows[0].keys())
        if missing:
            raise ValueError(f"{indices_path} is missing required columns: {sorted(missing)}")

        coords_xy = np.array([parse_coord(row["coord"]) for row in rows], dtype=np.int64)
        values = load_torch_tensor(preds_path)
        output_paths.append(
            marker_grid_to_ometiff(
                image_name,
                coords_xy,
                values,
                output_dir=output_dir,
                patch_size=patch_size,
                markers=markers,
                log1p_input=log1p_input,
            )
        )

    if not output_paths:
        raise FileNotFoundError(f"No *_preds.pt files found in {marker_dir}")
    return output_paths


def convert_legacy_tables(input_path, output_dir, markers, patch_size, infer_patch_size_flag, log1p_input):
    input_path = Path(input_path)
    csv_files = sorted(input_path.glob("*.csv")) if input_path.is_dir() else [input_path]
    for csv_path in csv_files:
        curr_patch_size = infer_patch_size(csv_path.name, patch_size) if infer_patch_size_flag else patch_size
        table_to_ometiff(
            csv_path,
            output_dir=output_dir,
            patch_size=curr_patch_size,
            markers=markers,
            log1p_input=log1p_input,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Convert VMISC inference outputs or patch-level marker CSV files to OME-TIFF."
    )
    parser.add_argument(
        "--input",
        default="marker/outputs",
        help="Input inference output directory, marker CSV file, or directory of marker CSV files.",
    )
    parser.add_argument("--output-dir", default="marker/outputs/ometiff", help="Directory for OME-TIFF outputs.")
    parser.add_argument("--markers", default=None, help="Optional marker CSV. Uses marker_name or first column.")
    parser.add_argument("--patch-size", type=int, default=507, help="Default patch size in source coordinates.")
    parser.add_argument(
        "--infer-patch-size",
        action="store_true",
        help="Use legacy slide-name rules for AG-3898 and -5M- patch sizes.",
    )
    parser.add_argument("--raw-input", action="store_true", help="Input marker values are not log1p transformed.")
    parser.add_argument(
        "--mode",
        choices=["auto", "inference-output", "marker-table"],
        default="auto",
        help="Input format. auto treats directories with 19plex/*_preds.pt as inference outputs.",
    )
    args = parser.parse_args()

    markers = read_markers(args.markers)
    input_path = Path(args.input)
    log1p_input = not args.raw_input
    is_inference_output = input_path.is_dir() and (input_path / "19plex").is_dir()

    if args.mode == "inference-output" or (args.mode == "auto" and is_inference_output):
        inference_output_to_ometiff(
            input_path,
            output_dir=args.output_dir,
            patch_size=args.patch_size,
            markers=markers,
            log1p_input=log1p_input,
        )
    else:
        convert_legacy_tables(
            input_path,
            args.output_dir,
            markers,
            args.patch_size,
            args.infer_patch_size,
            log1p_input,
        )


if __name__ == "__main__":
    main()
