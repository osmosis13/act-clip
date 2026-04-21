import torch
import clip
from typing import Dict, Any

class CLIPTextEncoder:
    def __init__(self, device='cuda'):
        self.device = device 
        self.model, _ = clip.load("ViT-B/32", device=self.device)
        self.model.eval()
    
    @torch.no_grad()
    def encode(self, text: str) -> torch.Tensor:
        """Encode text instruction to 512-dim embedding"""
        tokens = clip.tokenize([text], truncate=True).to(self.device)
        text_emb = self.model.encode_text(tokens)
        return text_emb.squeeze(0).float()  # [512]
    
    def batch_encode(self, texts: list) -> torch.Tensor:
        """Batch encode multiple instructions"""
        tokens = clip.tokenize(texts, truncate=True).to(self.device)
        text_embs = self.model.encode_text(tokens)
        return text_embs  # [N, 512]
