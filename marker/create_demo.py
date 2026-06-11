import argparse
import base64
import csv
import pickle
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def make_patch(rng, patch_idx, size=256):
    """Create a deterministic H&E-like RGB patch with mild structure."""
    yy, xx = np.mgrid[0:size, 0:size]
    phase = patch_idx * 0.17

    tissue = (
        0.45
        + 0.25 * np.sin((xx / 22.0) + phase)
        + 0.18 * np.cos((yy / 27.0) - phase)
        + rng.normal(0, 0.05, (size, size))
    )
    tissue = np.clip(tissue, 0, 1)

    rgb = np.empty((size, size, 3), dtype=np.uint8)
    rgb[..., 0] = np.clip(220 - 80 * tissue + rng.normal(0, 5, (size, size)), 0, 255)
    rgb[..., 1] = np.clip(175 - 70 * tissue + rng.normal(0, 5, (size, size)), 0, 255)
    rgb[..., 2] = np.clip(205 - 25 * tissue + rng.normal(0, 5, (size, size)), 0, 255)

    image = Image.fromarray(rgb, "RGB").filter(ImageFilter.GaussianBlur(radius=0.4))
    draw = ImageDraw.Draw(image, "RGBA")
    for _ in range(rng.integers(18, 34)):
        x = int(rng.integers(0, size))
        y = int(rng.integers(0, size))
        r = int(rng.integers(4, 12))
        color = (95, 55, 135, int(rng.integers(45, 105)))
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color)

    return image, tissue


def encode_jpeg(image):
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=88)
    return base64.urlsafe_b64encode(buffer.getvalue()).decode("ascii")


def marker_vector(tissue, rng, slide_idx, patch_idx):
    mean = float(tissue.mean())
    std = float(tissue.std())
    q90 = float(np.quantile(tissue, 0.90))
    base = np.linspace(0.2, 1.2, 19, dtype=np.float32)
    periodic = 0.25 * np.sin(np.arange(19, dtype=np.float32) * 0.7 + patch_idx * 0.11)
    slide_effect = 0.04 * slide_idx
    noise = rng.normal(0, 0.025, 19).astype(np.float32)
    values = 1.0 + base * mean + 0.7 * std + 0.25 * q90 + periodic + slide_effect + noise
    return np.clip(values, 0.001, None).astype(np.float32)


def write_tsv_with_index(path, rows):
    offsets = {}
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for idx, (image_b64, coord) in enumerate(rows):
            offsets[idx] = handle.tell()
            handle.write(f"{image_b64}\t{coord}\n")

    with path.with_suffix(path.suffix + ".index").open("wb") as handle:
        pickle.dump((len(rows), offsets), handle)


def build_demo_data(output_dir, patches, slides, seed):
    rng = np.random.default_rng(seed)
    output_dir = Path(output_dir)
    he_dir = output_dir / "he_patches"
    highplex_dir = output_dir / "highplex_ground_truth"
    csv_dir = output_dir / "csv_files"

    for directory in (he_dir, highplex_dir, csv_dir):
        directory.mkdir(parents=True, exist_ok=True)

    records = []
    rows_by_slide = {f"DEMO{i + 1:02d}": [] for i in range(slides)}

    for patch_idx in range(patches):
        slide_idx = patch_idx % slides
        img_id = f"DEMO{slide_idx + 1:02d}"
        x = (patch_idx // slides) * 256
        y = slide_idx * 256
        coord = f"x{x}_y{y}"

        image, tissue = make_patch(rng, patch_idx)
        rows_by_slide[img_id].append((encode_jpeg(image), coord))

        slide_highplex = highplex_dir / img_id
        slide_highplex.mkdir(exist_ok=True)
        np.save(slide_highplex / f"{img_id}_{coord}.npy", marker_vector(tissue, rng, slide_idx, patch_idx))

        records.append({"img_id": img_id, "coord": coord, "fold": patch_idx % 5})

    for img_id, rows in rows_by_slide.items():
        write_tsv_with_index(he_dir / f"{img_id}_patches.tsv", rows)

    with (csv_dir / "training_demo_100patches.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["img_id", "coord", "fold"])
        writer.writeheader()
        writer.writerows(records)

    return records


def main():
    parser = argparse.ArgumentParser(description="Create marker demo training data.")
    parser.add_argument("--output-dir", default="marker/demo_data")
    parser.add_argument("--patches", type=int, default=25)
    parser.add_argument("--slides", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260610)
    args = parser.parse_args()

    records = build_demo_data(args.output_dir, args.patches, args.slides, args.seed)
    print(f"Wrote {len(records)} demo patches to {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
