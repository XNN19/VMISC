# VMISC

VMISC predicts virtual multiplex immunofluorescence marker profiles from H&E whole-slide images and uses the resulting virtual spatial-proteomics features for colorectal-cancer prognosis modeling.

This folder is a compact public release in the same spirit as small model repositories such as MUSK: one README, one dependency file, portable configs, a demo notebook, and the smallest runnable code surface.

## News

- Consolidated WSI preprocessing into `preprocess/create_patches.py`.
- Kept the public workflows to marker training, marker inference, visualization, prognosis data prep, train, and test.
- Removed internal notes, duplicate model variants, and generated/private-data placeholders.

## Install

```bash
conda env create -f environment.yml
conda activate vmisc
```

or

```bash
pip install -r requirements.txt
pip install -e .
```

OpenSlide is required for WSI reading.

## Model And Data

Large files are intentionally excluded: slides, checkpoints, tensors, `.h5` patch files, OME-TIFFs, and clinical tables. Expected external inputs are:

- H&E WSIs readable by OpenSlide.
- CLAM-style coordinate `.h5` files from preprocessing.
- VMISC marker checkpoint `fold_0.pth`.
- Encoder weights for the selected pathology foundation model.
- Clinical CSVs with `case_id`, `slide_id`, `survival_months`, and `censorship`.

## Quick Start

Preprocess slides:

```bash
python preprocess/create_patches.py \
  --source data/slides \
  --save_dir outputs/preprocess \
  --preset argo.csv \
  --seg --patch --stitch
```

Run the marker smoke test:

```bash
python marker/create_demo.py
python marker/train.py --config marker/configs/demo.yaml --seed 123456 --devices 0 --splits 1
```

Train marker prediction:

```bash
python marker/train.py --config configs/examples/marker.yaml --seed 123456 --devices 0
```

Run WSI inference:

```bash
python marker/infer.py \
  --config configs/examples/marker.yaml \
  --checkpoint-dir outputs/marker/config_name/seed_123456/fold_0 \
  --output-dir outputs/vmisc_inference \
  --he-dir data/slides \
  --patch-dir outputs/preprocess/patches \
  --ext .svs \
  --accelerator gpu \
  --devices 0
```

Convert marker outputs to OME-TIFF:

```bash
python visualize.py --input outputs/vmisc_inference --output-dir outputs/vmisc_inference/ometiff
```

Prepare and train the prognosis model:

```bash
python prognosis/prepare_data.py \
  --outputs-dir outputs/vmisc_inference \
  --output-root data/prognosis \
  --name vmisc \
  --clinical-csv data/clinical/prognosis.csv

python prognosis/train.py --config configs/examples/prognosis.yaml --seed_list 1
```

Evaluate prognosis:

```bash
python prognosis/test.py \
  --config configs/examples/prognosis.yaml \
  --checkpoint results/prognosis/<run_name>/s_<split>_checkpoint_0.pt \
  --output-dir results/prognosis_test
```

## Layout

```text
vmisc-pull/
|-- configs/examples/        # public YAML templates
|-- marker/                  # marker prediction train/infer/demo code
|-- notebooks/               # demo/data-contract notebooks
|-- preprocess/              # WSI tiling utilities
|-- prognosis/               # prognosis prep/train/test code
|-- visualize.py             # OME-TIFF export
|-- environment.yml
|-- requirements.txt
|-- pyproject.toml
`-- README.md
```

## Data Contracts

Marker training manifests use `img_id`, `coord`, and `fold`.

Marker inference writes:

```text
19plex/<slide_id>_preds.pt
19plex/<slide_id>_indices.csv
19plex_feat/<slide_id>_preds.pt
static/<slide_id>_preds.pt
```

Prognosis clinical CSVs use `case_id`, `slide_id`, `survival_months`, and `censorship`.
