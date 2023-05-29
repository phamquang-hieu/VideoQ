import torch
from torch import nn
from collections import OrderedDict
from timm.models.layers import trunc_normal_
import sys
from prompt import PromptPool
sys.path.append("../")
from clip.model import QuickGELU


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MultiframeIntegrationTransformer(nn.Module):
    def __init__(self, T, embed_dim=512, layers=1, pool_size=25, pool_use_freq=False, pool_prompts_per_sample=5, pool_prompt_length=8):
        super().__init__()
        self.T = T
        transformer_heads = embed_dim // 64
        self.positional_embedding = nn.Parameter(torch.empty(1, T, embed_dim))
        trunc_normal_(self.positional_embedding, std=0.02)
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(d_model=embed_dim, n_head=transformer_heads) for _ in range(layers)])

        self.pool_prompts_per_sample = pool_prompts_per_sample
        self.pool_prompt_length = pool_prompt_length
        self.pool_size = pool_size
        self.prompt_pool = PromptPool(pool_size=pool_size,
                                      embedd_dim=embed_dim,
                                      use_freq=pool_use_freq, 
                                      pool_prompts_per_sample=pool_prompts_per_sample, 
                                      pool_prompt_length=pool_prompt_length)

        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Linear,)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(self, x):
        b, t, d = x.shape
        ori_x = x
        x = x + self.positional_embedding

        prompt_key_loss = None
        if self.pool_size > 0:
            prompted_text = torch.zeros(b, t + self.pool_prompt_length * self.pool_prompts_per_sample, d).to(x.device)
            prompts, prompt_key_loss = self.prompt_pool(x.mean(dim=1).unsqueeze(dim=1))

            for i in range(b):
                prompted_text[i] = torch.concat(x[i], prompts[i], dim=0)
            x = prompted_text

        x = x.permute(1, 0, 2)
        x = self.resblocks(x)
        x = x.permute(1, 0, 2)  
        x = x.type(ori_x.dtype) + ori_x
        
        return x.mean(dim=1, keepdim=False), prompt_key_loss
