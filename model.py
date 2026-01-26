import math
from typing import Any, Callable, Optional, Tuple, Type, Sequence, Union
import torch
import torch.nn as nn
from einops import rearrange
from utils.math_utils import get_2d_sincos_pos_embed

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
            
        embeddings = self.embedding_table(labels)
        return embeddings


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding """
    def __init__(self, img_size, patch_size, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x) # (B, E, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, N, E)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        image_size=256,
        image_channels=3,
        patch_size=2,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        out_channels=3,
        class_dropout_prob=0.1,
        num_classes=1000,
        dropout=0.0,
        ignore_dt=False,
    ):
        super().__init__()
        self.image_size = image_size
        self.image_channels = image_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.out_channels = out_channels
        self.class_dropout_prob = class_dropout_prob
        self.num_classes = num_classes
        self.ignore_dt = ignore_dt
        self.dropout = dropout
        
        # 1. Input Adapters
        self.patch_embed = PatchEmbed(image_size, patch_size, image_channels, hidden_size)
        self.x_embedder = self.patch_embed # alias
        
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.dt_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        
        # 2. Positional Embedding (Fixed, Sinusoidal)
        num_patches = self.patch_embed.num_patches
        # Note: We rely on get_2d_sincos_pos_embed returning a tensor of shape (num_patches, hidden_size)
        # We register it as a buffer so it saves with the model but isn't updated by optimizer
        pos_embed = get_2d_sincos_pos_embed(embed_dim=hidden_size, grid_size=int(num_patches**0.5))
        self.register_buffer('pos_embed', torch.from_numpy(pos_embed).float().unsqueeze(0), persistent=False)

        # 3. Transformer Blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        
        # 4. Final Layer
        self.final_layer = FinalLayer(hidden_size, patch_size, out_channels)

        # 5. Log Variance (Optional, from JAX code)
        # Assuming simple scalar or learned scalar per step. 
        # JAX code: nn.Embed(256, 1)
        self.logvar_embed = nn.Embedding(256, 1)
        nn.init.constant_(self.logvar_embed.weight, 0.0)


    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, dt, y, train=False, return_activations=False):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latents)
        t: (N,) tensor of diffusion timesteps
        dt: (N,) tensor of delta timesteps
        y: (N,) tensor of class labels
        """
        # 1. Patchify & Embed
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H*W / P**2
        
        # 2. Condition Embedding
        t_emb = self.t_embedder(t)
        
        if self.ignore_dt:
            dt_emb = 0
        else:
            dt_emb = self.dt_embedder(dt)
            
        y_emb = self.y_embedder(y, self.training)
        
        # Sum conditions
        c = t_emb + dt_emb + y_emb ## TODO: Check if JAX code does element-wise sum or concat. Code says: c = te + ye + dte
        
        # 3. Transformer Blocks
        for block in self.blocks:
            x = block(x, c)
        
        # 4. Final Layer
        x = self.final_layer(x, c)
        
        # 5. Unpatchify
        x = self.unpatchify(x)
        
        # 6. Logvars (Auxiliary output)
        if hasattr(self, 'logvar_embed'):
            t_discrete = (t * 255).long().clamp(0, 255)
            logvars = self.logvar_embed(t_discrete) * 100
        else:
            logvars = torch.zeros_like(x)

        if return_activations:
            return x, logvars, {} # Returning empty dict for activations for now
            
        return x


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        # Attention - Using standard MHSA to start, can optimize later
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        
        # Approximate GELU for consistency with JAX/Flax defaults
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=lambda: nn.GELU(approximate="tanh"), drop=0)
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        
        # 1. Attention Block
        # Modulate
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        
        # Attention (PyTorch MHA)
        # Note: MHA handles q, k, v projections internally
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        
        # Residual + Gate
        x = x + gate_msa.unsqueeze(1) * attn_out
        
        # 2. MLP Block
        # Modulate
        x_norm2 = modulate(self.norm2(x), shift_mlp, scale_mlp)
        
        # MLP
        mlp_out = self.mlp(x_norm2)
        
        # Residual + Gate
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        
        return x

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

        # Zero Init Logic (Crucial!)
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x
