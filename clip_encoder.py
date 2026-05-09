import torch
import clip
import torch.nn as nn

class CLIPDualEncoder(nn.Module):
    def __init__(self, device=None, freeze=True, unfreeze_last_n_blocks=0):
        super().__init__()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.model.eval()
        self.image_dim = 512
        self.text_dim  = 512
        self.patch_size = 32

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        if unfreeze_last_n_blocks > 0:
            # Convert entire visual encoder to float32 FIRST
            # — float16 gradients overflow (max 65504), causing NaN on first backward
            self.model.float()

            total_blocks = len(self.model.visual.transformer.resblocks)
            for i in range(total_blocks - unfreeze_last_n_blocks, total_blocks):
                for param in self.model.visual.transformer.resblocks[i].parameters():
                    param.requires_grad = True
            self.model.visual.ln_post.weight.requires_grad = True
            self.model.visual.ln_post.bias.requires_grad = True
            if self.model.visual.proj is not None:
                self.model.visual.proj.requires_grad = True

    def encode_text(self, text: str) -> torch.Tensor:  # no @torch.no_grad()
        tokens   = clip.tokenize([text], truncate=True).to(self.device)
        text_emb = self.model.encode_text(tokens)
        return text_emb.squeeze(0).float()

    def encode_image_patches(self, image: torch.Tensor) -> torch.Tensor:
        visual = self.model.visual

        # If visual encoder is float32 (unfrozen), image stays float32
        # If float16 (frozen), cast to match weights as before
        target_dtype = visual.conv1.weight.dtype
        image = image.to(dtype=target_dtype)

        x = visual.conv1(image)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        cls = visual.class_embedding.unsqueeze(0).unsqueeze(0).expand(x.shape[0], -1, -1)
        x   = torch.cat([cls, x], dim=1)
        x   = x + visual.positional_embedding
        x   = visual.ln_pre(x)
        x   = x.permute(1, 0, 2)
        x   = visual.transformer(x)
        x   = x.permute(1, 0, 2)
        x   = visual.ln_post(x[:, 1:, :])

        if visual.proj is not None:
            x = x @ visual.proj

        return x.float()