import timm
from timm.layers import SwiGLUPacked

import torch
import pytorch_lightning as pl


class Virchow2Lightning(pl.LightningModule):
    def __init__(self, save_path=None):
        super().__init__()
        self.save_path = save_path
        
        self.model = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
        for m in self.model.parameters():
            m.requires_grad = False
        
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        data = batch[0]
        img_id, coord = batch[2], batch[3]
        with torch.inference_mode():
            preds = self(data)
        class_token = preds[:, 0]
        patch_tokens = preds[:, 5:]
        embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)  # size: 1 x 2560
        return embedding
    