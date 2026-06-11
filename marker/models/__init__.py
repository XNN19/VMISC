import pytorch_lightning as pl
import importlib
import torch
from .model_utils import get_rank

from .virchow_reg import VirchowReg

from torchmetrics.functional import r2_score
from scipy.stats import pearsonr
from torchmetrics.functional import symmetric_mean_absolute_percentage_error



def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


class PlexModel(pl.LightningModule):
    def __init__(self, config, save_path, encoder=None):
        super().__init__()
        self.config = config
        self.save_path = save_path

        self.encoder = encoder
        if self.encoder is not None:
            for m in self.encoder.parameters():
                m.requires_grad = False
            self.encoder.eval()

        self.criterion = get_obj_from_str(
            self.config["Loss"]["name"]
            )()

        self.model = get_obj_from_str(config["Model"]["name"])(**config["Model"]["params"])

        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.pred_outputs = []

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, batch):
        result_dict = self(batch)

        logits = result_dict['logit']
        targets = result_dict['target']
        
        loss = self.criterion(logits, targets)
        error = symmetric_mean_absolute_percentage_error(logits, targets)
        
        metrics = {"loss": loss, "error": error, 'logit_target_dict': result_dict}
        
        return metrics


    def training_step(self, batch, batch_idx):
        return self.compute_loss(batch)["loss"]

    def on_train_epoch_end(self):
        self.lr_scheduler.step()


    def eval_epoch(self, mode='eval'):

        step_outputs = self.validation_step_outputs if mode == 'eval' else self.test_step_outputs

        all_logits = torch.cat([out['logit_target_dict']['logit'] for out in step_outputs], dim=0)
        all_targets = torch.cat([out['logit_target_dict']['target'] for out in step_outputs], dim=0)

        all_loss = self.criterion(all_logits, all_targets)
        all_errors = symmetric_mean_absolute_percentage_error(all_logits, all_targets)
        
        self.log('val_loss', all_loss, sync_dist=True)
        self.log('val_err', all_errors, sync_dist=True)
        
        # total loss
        total_loss = sum([out['loss'] for out in step_outputs]) / len(step_outputs)
        total_error = sum([out['error'] for out in step_outputs]) / len(step_outputs)
        self.log('val_loss_total', total_loss, sync_dist=True)
        self.log('val_err_total', total_error, sync_dist=True)

        if mode == 'test':
            
            all_preds = all_logits.cpu().numpy()
            all_targets = all_targets.cpu().numpy()
            
            r2_scores = []
            pccs = []
            ppval = []
            for i in range(all_preds.shape[1]):
                test_r2 = r2_score(torch.FloatTensor(all_preds[:, i]), torch.FloatTensor(all_targets[:, i]))
                test_pcc, test_ppval = pearsonr(all_preds[:, i].ravel(), all_targets[:, i].ravel())
                r2_scores.append(test_r2)
                pccs.append(test_pcc)
                ppval.append(test_ppval)
        
        if get_rank() == 0:
            if mode == 'test':
                print(f'\nTest R2 for each channel: {r2_scores} \
                      \nTest PCC for each channel: {pccs} with p-value: {ppval}\n')

            print(f"\nval_error: {all_errors:.4f}, val_loss: {all_loss: .4f}, \
                    val_error_total: {total_loss: .4f}, val_loss_total: {total_error: .4f}\n")


    def validation_step(self, batch, batch_idx):
        # Compute loss and metrics for the current validation batch
        with torch.inference_mode():
            outputs = self.compute_loss(batch)
        self.validation_step_outputs.append(outputs)


    def on_validation_epoch_end(self):
        self.eval_epoch(mode='eval')
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        with torch.inference_mode():
            ret = self.compute_loss(batch)
        self.test_step_outputs.append(ret)
        return ret

    def on_test_epoch_end(self):
        self.eval_epoch(mode='test')
        self.test_step_outputs.clear()


    def configure_optimizers(self):
        conf_optim = self.config["Optimizer"]
        name = conf_optim["optimizer"]["name"]
        optimizer_cls = getattr(torch.optim, name)
        scheduler_cls = getattr(torch.optim.lr_scheduler, conf_optim["lr_scheduler"]["name"])

        optim = optimizer_cls(filter(lambda p: p.requires_grad, self.parameters()), **conf_optim["optimizer"]["params"])
        self.lr_scheduler = scheduler_cls(optim, **conf_optim["lr_scheduler"]["params"])
        return optim
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        data = batch[:2]
        img_id, coord = batch[2], batch[3]
        with torch.inference_mode():
            preds = self(data)
        self.pred_outputs.append([preds, img_id, coord])
        
        return preds

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
