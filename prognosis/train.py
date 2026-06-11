import argparse
import os
from timeit import default_timer as timer
import warnings
import yaml
import random

from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import torch

from dataset_mcat import MCAT_Survival_Dataset
from core_utils import train


def seed_everything(seed: int = 1) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
### Sets Seed for reproducible experiments.

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"YAML at {path} did not parse to a dict.")
    return cfg

def build_dsets(cfg):
    for k in ["csv_path", "data_dir_path", "data_dir_omic"]:
        if k not in cfg:
            raise KeyError(f"Missing required key in YAML: {k}")

    train_csv = str(cfg["csv_path"])
    val_csv = str(cfg.get("val_csv_path") or train_csv.replace("/train/", "/val/"))
    print(f'train_csv: {train_csv}')
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    # NOTE: dataset_mcat.MCAT_Survival_Dataset signature supports these keys
    he_suffix = str(cfg.get("he_suffix", ".pt"))
    omic_suffix = str(cfg.get("omic_suffix", "_feats.pt"))
    n_bins = int(cfg.get("n_classes", 4))  # In your YAML this is 1; keep as-is unless you want 4.
    label_col = str(cfg.get("label_col", "survival_months"))

    train_set = MCAT_Survival_Dataset(
        df=train_df,
        data_dir_path=str(cfg["data_dir_path"]),
        data_dir_omic=str(cfg["data_dir_omic"]),
        label_col=label_col,
        n_bins=n_bins,
        he_suffix=he_suffix,
        omic_suffix=omic_suffix,
    )
    val_set = MCAT_Survival_Dataset(
        df=val_df,
        data_dir_path=str(cfg["data_dir_path"]),
        data_dir_omic=str(cfg["data_dir_omic"]),
        label_col=label_col,
        n_bins=n_bins,
        he_suffix=he_suffix,
        omic_suffix=omic_suffix,
    )
    return train_set, val_set, os.path.basename(train_csv).split('.')[0]
warnings.filterwarnings("ignore", category=FutureWarning)


def main(args):
    base_results_dir = args.results_dir

    seed = int(getattr(args, "seed", 1))

    cfg = load_yaml(args.config)
        
    print(f'seed: {seed}')
    seed_everything(seed)
    train_dataset, val_dataset, split_name = build_dsets(cfg)
            
    split_name = split_name + f'_seed{seed}'
            
    args.results_dir = os.path.join(base_results_dir, split_name)
    os.makedirs(args.results_dir, exist_ok=True)
            
    start_time = timer()
        
            
    print(f'training: {len(train_dataset)}, validation: {len(val_dataset)}')
    datasets = (train_dataset, val_dataset)

    if args.task_type == 'survival':
        val_latest, cindex_all_i, cindex_all_mean_i, cindex_all_max_i = train(datasets, split_name, args)
        print(cindex_all_i)
        print(cindex_all_mean_i)
        print(cindex_all_max_i)

    end_time = timer()
    print(f'Split {split_name} Time: {end_time - start_time:.4f} seconds')
    

parser = argparse.ArgumentParser(description='Configurations for Survival Analysis on TCGA Data.')

parser.add_argument("--config", type=str, required=True, help="Path to YAML config (e.g. n2_argo.yaml)")
parser.add_argument('--k', type=int, default=5, help='Number of folds (default: 5)')
parser.add_argument('--k_start', type=int, default=-1, help='Start fold (Default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='End fold (Default: -1, first fold)')
parser.add_argument('--results_dir', type=str, default='./results', help='Results directory (Default: ./results)')
parser.add_argument('--which_splits', type=str, default='5foldcv',
                    help='Which splits folder to use in ./splits/ (Default: ./splits/5foldcv')
parser.add_argument('--split_dir', type=str, default='tcga_blca_100',
                    help='Which cancer type within ./splits/<which_splits> to use for training. Used synonymously for "task" (Default: tcga_blca_100)')
parser.add_argument('--log_data', action='store_true', default=True, help='Log data using tensorboard')
parser.add_argument('--overwrite', action='store_true', default=False,
                    help='Whether or not to overwrite experiments (if already ran)')

### Model Parameters.
parser.add_argument('--model_type', type=str, choices=['snn', 'deepset', 'amil', 'mi_fcn', 'mcat'], default='mcat',
                    help='Type of model (Default: mcat)')
parser.add_argument('--mode', type=str, choices=['omic', 'path', 'pathomic', 'cluster', 'coattn'], default='coattn',
                    help='Specifies which modalities to use / collate function in dataloader.')
parser.add_argument('--fusion', type=str, choices=['None', 'concat', 'bilinear'], default='concat',
                    help='Type of fusion. (Default: concat).')
parser.add_argument('--apply_sig', action='store_true', default=False,
                    help='Use genomic features as signature embeddings.')
parser.add_argument('--apply_sigfeats', action='store_true', default=False,
                    help='Use genomic features as tabular features.')
parser.add_argument('--drop_out', action='store_true', default=True, help='Enable dropout (p=0.25)')
parser.add_argument('--model_size_wsi', type=str, default='small', help='Network size of AMIL model')
parser.add_argument('--model_size_omic', type=str, default='small', help='Network size of SNN model')

### Optimizer Parameters + Survival Loss Function
parser.add_argument('--opt', type=str, choices=['adam', 'sgd'], default='adam')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size (Default: 1, due to varying bag sizes)')
parser.add_argument('--gc', type=int, default=32, help='Gradient Accumulation Step.')
parser.add_argument('--max_epochs', type=int, default=20, help='Maximum number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate (default: 0.0001)')
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce', 'ce_surv', 'nll_surv', 'cox_surv'],
                    default='nll_surv', help='slide-level classification loss function (default: ce)')
parser.add_argument('--label_frac', type=float, default=1.0, help='fraction of training labels (default: 1.0)')
parser.add_argument('--bag_weight', type=float, default=0.7,
                    help='clam: weight coefficient for bag-level loss (default: 0.7)')
parser.add_argument('--reg', type=float, default=1e-5, help='L2-regularization weight decay (default: 1e-5)')
parser.add_argument('--alpha_surv', type=float, default=0.0, help='How much to weigh uncensored patients')
parser.add_argument('--reg_type', type=str, choices=['None', 'omic', 'pathomic'], default='None',
                    help='Which network submodules to apply L1-Regularization (default: None)')
parser.add_argument('--lambda_reg', type=float, default=1e-4, help='L1-Regularization Strength (Default 1e-4)')
parser.add_argument('--weighted_sample', action='store_true', default=False, help='Enable weighted sampling')
parser.add_argument('--early_stopping', action='store_true', default=False, help='Enable early stopping')
parser.add_argument('--testing', action='store_true', default=False, help='Enable early stopping')
parser.add_argument('--fast_dev_run', action='store_true', default=False,
                    help='Run a minimal development pass when supported by downstream code.')



args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(args.config, 'r') as f:
    config = yaml.safe_load(f)
    for key, value in config.items():
        if hasattr(args, key):
            if key == 'fast_dev_run' and args.fast_dev_run:
                continue
            setattr(args, key, value)
        else:
            setattr(args, key, value)

args.n_classes = config.get("n_classes", 2)
args.max_epochs = config.get("max_epochs", 20)
args.task_type = 'survival'
settings = {'num_splits': args.k,
            'k_start': args.k_start,
            'k_end': args.k_end,
            # 'task': args.task,
            'max_epochs': args.max_epochs,
            'results_dir': args.results_dir,
            'lr': args.lr,
            # 'experiment': args.exp_code,
            'reg': args.reg,
            'label_frac': args.label_frac,
            # 'inst_loss': args.inst_loss,
            'bag_loss': args.bag_loss,
            'bag_weight': args.bag_weight,
            'model_type': args.model_type,
            'model_size_wsi': args.model_size_wsi,
            'model_size_omic': args.model_size_omic,
            "use_drop_out": args.drop_out,
            'weighted_sample': args.weighted_sample,
            'gc': args.gc,
            'opt': args.opt}
print('\nLoad Dataset')

if not os.path.isdir(args.results_dir):

    os.makedirs(args.results_dir, exist_ok=True)

print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))

if __name__ == "__main__":
    start = timer()
    results = main(args)
    end = timer()
    print("finished!")
    print("end script")
    print('Script Time: %f seconds' % (end - start))
