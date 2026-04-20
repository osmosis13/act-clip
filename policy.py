import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
from clip_encoder import CLIPTextEncoder

from detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer
import IPython
e = IPython.embed

class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']

        # CLIP integration
        self.clip_encoder = CLIPTextEncoder()
        self.text_dim = 512
        
        # Get hidden_dim from DETR model 
        self.hidden_dim = 256 
        self.text_proj = nn.Linear(self.text_dim, self.hidden_dim)

        print(f'KL Weight {self.kl_weight}')

    def __call__(self, qpos, image, actions=None, is_pad=None, instruction=None):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        
        if actions is not None:  # TRAINING
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]
            
            # instruction is already an embedding tensor [B, 512]
            if instruction is not None:
                # instruction: [B, 512] from dataset
                text_emb = instruction.to(actions.device).float()      # [B, 512]
                text_emb = self.text_proj(text_emb)                    # [B, 256] (hidden_dim)
            else:
                text_emb = None

            a_hat, is_pad_hat, (mu, logvar) = self.model(
                qpos, image, env_state, actions, is_pad, text_emb=text_emb
                )
            
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            
            loss_dict = {
                'l1': l1,
                'kl': total_kld[0],
                'loss': l1 + total_kld[0] * self.kl_weight
            }
            return loss_dict
        
        else:
            # At inference you have raw text, not embeddings
            if instruction is not None:
                # instruction is a string like "transfer cube to left gripper"
                instr_emb = self.clip_encoder.encode_single(instruction)  # [512]
                instr_emb = instr_emb.unsqueeze(0).to(image.device).float()  # [1, 512]
                text_emb = self.text_proj(instr_emb)                         # [1, 256]
            else:
                text_emb = None

            a_hat, _, (_, _) = self.model(
                qpos, image, env_state, text_emb=text_emb
                )
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


class CNNMLPPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model # decoder
        self.optimizer = optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None # TODO
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            loss_dict = dict()
            loss_dict['mse'] = mse
            loss_dict['loss'] = loss_dict['mse']
            return loss_dict
        else: # inference time
            a_hat = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld
