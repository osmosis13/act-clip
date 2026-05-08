import torch
import clip
import torch.nn as nn

class CLIPDualEncoder(nn.Module):
    def __init__(self, device=None, freeze=True):
        super().__init__()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.model.eval()

        # ViT-B/32 image encoder output
        self.image_dim = 512    # CLIP joint embedding dim
        self.text_dim  = 512
        self.patch_size = 32    # ViT-B/32 divides image into 32x32 patches
        
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    @torch.no_grad()
    def encode_text(self, text: str) -> torch.Tensor:
        tokens   = clip.tokenize([text], truncate=True).to(self.device)
        text_emb = self.model.encode_text(tokens)
        return text_emb.squeeze(0).float()                    # [512]

    def encode_image_patches(self, image: torch.Tensor) -> torch.Tensor:
        """
        Returns patch-level features from CLIP ViT, not the pooled CLS token.
        image: [B, 3, 224, 224] — must be resized to 224x224 for ViT-B/32
        returns: [B, num_patches, 512] where num_patches = 196 for 224x224
        """
        # Hook into the ViT to get intermediate patch tokens
        # rather than the final pooled embedding
        visual = self.model.visual

        # Convert input to match CLIP's weight dtype (float16 on CUDA)
        image = image.to(dtype=visual.conv1.weight.dtype) 
        
        x = visual.conv1(image)                               # [B, 768, 7, 7]
        x = x.reshape(x.shape[0], x.shape[1], -1)            # [B, 768, 49]
        x = x.permute(0, 2, 1)                               # [B, 49, 768]
        
        # Prepend CLS token
        cls = visual.class_embedding.unsqueeze(0).unsqueeze(0)
        cls = cls.expand(x.shape[0], -1, -1)                 # [B, 1, 768]
        x   = torch.cat([cls, x], dim=1)                     # [B, 50, 768]
        
        x = x + visual.positional_embedding                  # add pos encoding
        x = visual.ln_pre(x)
        x = x.to(dtype=visual.conv1.weight.dtype)            # ensure same dtype for transformer
        x = x.permute(1, 0, 2)                               # [seq, B, 768]
        x = visual.transformer(x)
        x = x.permute(1, 0, 2)                               # [B, seq, 768]
        x = visual.ln_post(x[:, 1:, :])                      # drop CLS, [B, 49, 768]
        
        if visual.proj is not None:
            x = x @ visual.proj                               # [B, 49, 512]
        
        return x.float()