import torch
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
        self.log_std = nn.Parameter(
            torch.zeros(args_override.get('state_dim', 14))
        )

        # CLIP integration
        self.clip_encoder = CLIPTextEncoder()
        self.text_dim = 512
        
        # Get hidden_dim from DETR model 
        self.hidden_dim = args_override['hidden_dim']
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
                text_emb = instruction.to(actions.device).float()
                text_emb = torch.nn.functional.normalize(text_emb, dim=-1)
                text_emb = self.text_proj(text_emb)          # [B, hidden_dim]
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
                # instruction is a string at inference
                if isinstance(instruction, list):
                    instruction = instruction[0]  # ["transfer cube"] -> "transfer cube"
                instr_emb = self.clip_encoder.encode(instruction)        # <-- was encode_single
                instr_emb = instr_emb.unsqueeze(0).to(image.device).float()  # [1, 512]
                text_emb = self.text_proj(instr_emb)                         # [1, 512]
            else:
                text_emb = None

            a_hat, _, (_, _) = self.model(
                qpos, image, env_state, text_emb=text_emb
                )
            return a_hat
        
    def forward_rl(self, qpos, image, instruction=None):
        """
        Returns sampled action (no grad) and log_prob (with grad).
        Memory-efficient version for RL rollouts.
        """
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        image = normalize(image)

        if instruction is not None:
            if isinstance(instruction, list):
                instruction = instruction[0]
            instr_emb = self.clip_encoder.encode(instruction)
            instr_emb = instr_emb.unsqueeze(0).to(image.device).float()
            text_emb = self.text_proj(instr_emb)
        else:
            text_emb = None

        # Forward pass to get mean action
        a_mean, _, (_, _) = self.model(qpos, image, env_state, text_emb=text_emb)

        # Sample action with gradient ONLY through log_std (not through the heavy CNN/transformer)
        std = self.log_std.exp().unsqueeze(0).unsqueeze(0)
        
        # Detach mean from the computation graph — we only need log_prob gradient through log_std
        dist = torch.distributions.Normal(a_mean.detach(), std)
        a_sample = dist.sample()
        log_prob = dist.log_prob(a_sample).sum(dim=-1)

        return a_sample, log_prob

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
