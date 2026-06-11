
import timm
from timm.layers import SwiGLUPacked
import torch.nn as nn
import torch

class RegGatedAttnHead(nn.Module):
    def __init__(self, input_size=2560, hidden_size=512, output_size=19):
        super(RegGatedAttnHead, self).__init__()
        self.attn_value = [
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
        ]
        self.attn_gate = [
            nn.Linear(input_size, hidden_size),
            nn.Sigmoid(),
        ]
        self.attn_value = nn.Sequential(*self.attn_value)
        self.attn_gate = nn.Sequential(*self.attn_gate)

        self.attn = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        value = self.attn_value(x)
        gate = self.attn_gate(x)
        output_feat = value.mul(gate)
        output = self.attn(output_feat)
        return output, output_feat


class VirchowReg(nn.Module):
    def __init__(self, is_embed, is_unfrozen, reghead_param):
        super(VirchowReg, self).__init__()
        
        self.is_embed = is_embed
        
        if is_embed:
            self.backbone = nn.Identity()
        else:
            model = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
        
            for m in model.parameters():
                m.requires_grad = False
            
            if is_unfrozen:
                for m in model.blocks[-3:].parameters():
                    m.requires_grad = True
            
            self.backbone = model
        
        self.header = RegGatedAttnHead(**reghead_param)

    def forward(self, batch):
        he, highplex = batch
        embedding = self.backbone(he)
        
        if self.is_embed:
            pass
        else:
            class_token = embedding[:, 0]
            patch_tokens = embedding[:, 5:]
            embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
        
        output, output_feat = self.header(embedding)
        outdict = {'logit': output, 'target': highplex, 'embedding': output_feat}
        return outdict