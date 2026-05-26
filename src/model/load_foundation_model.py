import torch
from src.model.moge.v2 import MoGeModel

def load_foundation_model(cfg):
    vggt = None
    # Only load foundation models during training when depth loss is active.
    # During eval (mode=="test"), depth/normal pseudo-labels are never computed.
    if cfg.mode == "train" and (cfg.train.depth_loss > 0 or cfg.train.normal_loss > 0):
        if cfg.train.reproj_model=='vggt':   
            from src.model.encoder.backbone.vggt.vggt import VGGT
            
            vggt = VGGT()
            msg = vggt.load_state_dict(torch.hub.load_state_dict_from_url(
                "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt",))
            vggt.eval()
            for param in vggt.parameters():
                param.requires_grad = False
        elif cfg.train.reproj_model=='moge':
            vggt = MoGeModel.from_pretrained('Ruicheng/moge-2-vitl-normal')
            vggt.eval()
            for param in vggt.parameters():
                param.requires_grad = False
        
    return vggt