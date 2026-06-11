import pytorch_lightning as pl
import torch.cuda
from pytorch_lightning.callbacks import ModelCheckpoint
from wsi_dataset import WSIDataModule
import yaml

from models import PlexModel
import os
import random
import shutil
import numpy as np
import argparse

def read_yaml(fpath):
    with open(fpath, "r", encoding="utf-8") as file:
        return dict(yaml.safe_load(file))

def fix_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)  # torch >= 1.8


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VMISC marker prediction models.")
    parser.add_argument("--config", default=os.path.join("configs", "examples", "marker.yaml"),
                        help="Path to a VMISC marker-prediction YAML config.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override General.seed from the YAML config.")
    parser.add_argument("--devices", nargs="+", type=int, default=None,
                        help="GPU device ids passed to PyTorch Lightning, for example --devices 0.")
    parser.add_argument("--splits", type=int, default=1, help="Number of folds/splits to train.")
    parser.add_argument("--output-root", default="workspace/models",
                        help="Directory where exported fold state_dict files are stored.")
    args = parser.parse_args()

    config_path = args.config
    fname = os.path.splitext(os.path.basename(config_path))[0]

    config_yaml = read_yaml(config_path)
    for key, value in config_yaml.items():
        print(f"{key.ljust(30)}: {value}")
    
    num_gpus = args.devices if args.devices is not None else config_yaml['General'].get('num_gpus', [0])
    dist = False
    N_SPLITS = args.splits

    seed = args.seed if args.seed is not None else config_yaml['General'].get('seed', 123456)
    
    proj = config_yaml['Data']['dataframe']
    proj = proj.split('/')[-1].split('.')[0]
    
    torch.set_float32_matmul_precision('medium')
    
    fix_seed(seed)
    workspace = f'outputs_{fname}_seed_{seed}'
    rets_fold = []                  # store the best performance results of each fold
        
    for split_k in range(N_SPLITS):

            dm = WSIDataModule(config_yaml, split_k=split_k, dist=dist)
            resume_path = None
            
            save_path = f"{args.output_root}/{fname}/{proj}/seed_{seed}/fold_{split_k}/"
            os.makedirs(save_path, exist_ok=True)

            model = PlexModel(config_yaml, save_path=save_path)
            
            save_fname = '{epoch}-{val_loss:.4f}'
            monitor = 'val_loss'

            checkpoint_callback = ModelCheckpoint(
                monitor=monitor,
                dirpath=workspace,
                filename=save_fname,
                save_top_k=1,
                mode='min',
                verbose=True,
            )
            
            # trainer
            trainer = pl.Trainer(
                accelerator='gpu',
                devices=num_gpus,
                deterministic=False,
                precision=16,
                callbacks=[checkpoint_callback],
                max_epochs=config_yaml['General']['epochs'],
                accumulate_grad_batches=config_yaml['General']['acc_steps'],
                logger=False,
            )

            trainer.fit(model, datamodule=dm, ckpt_path=resume_path)

            wts = trainer.checkpoint_callback.best_model_path
            trainer.test(model, datamodule=dm, ckpt_path=wts)
            torch.save(torch.load(wts)['state_dict'], f"{save_path}/fold_{split_k}.pth")
    shutil.rmtree(workspace, ignore_errors=True)
