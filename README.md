# VMISC

VMISC predicts virtual multiplex immunofluorescence marker profiles from H&E whole-slide images and uses the resulting virtual spatial-proteomics features for colorectal-cancer prognosis modeling.


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

## Quick Start

### WSI preprocess

Preprocess slides:

```bash
python preprocess/create_patches.py \
  --source data/slides \
  --save_dir outputs/preprocess \
  --preset argo.csv \
  --seg --patch --stitch
```

### V-MISC train

Run the marker smoke test:

```bash
python marker/create_demo.py
python marker/train.py --config marker/configs/demo.yaml --seed 123456 --devices 0 --splits 1
```

Train marker prediction:

```bash
python marker/train.py --config configs/examples/marker.yaml --seed 123456 --devices 0
```

### Inference highplex from WSI

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

**Convert marker outputs to OME-TIFF**:

```bash
python visualize.py --input outputs/vmisc_inference --output-dir outputs/vmisc_inference/ometiff
```

### Clinical multimodal prognosis model

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
