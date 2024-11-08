from timm.models.layers import trunc_normal_
import torch
from torch import nn
import sys
sys.path.append("../")
from clip.model import QuickGELU


class MulitHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)


        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        B, N, C = q.shape
        B, M, C = k.shape
        q = self.q_proj(q).reshape(B, N, self.num_heads, C // self.num_heads).permute(0,2,1,3)
        k = self.k_proj(k).reshape(B, M, self.num_heads, C // self.num_heads).permute(0,2,1,3)
        v = self.v_proj(v).reshape(B, M, self.num_heads, C // self.num_heads).permute(0,2,1,3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PromptGeneratorLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.,
    ):
        super().__init__()
        self.cross_attn = MulitHeadAttention(d_model, nhead, proj_drop=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            QuickGELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x, visual):
        q = k = v = self.norm1(x)
        x = x + self.cross_attn(q, visual, visual)
        x = x + self.dropout(self.mlp(self.norm3(x)))
        return x


class VideoSpecificPrompt(nn.Module):
    def __init__(self, layers=2, embed_dim=512, alpha=0.1,):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.decoder = nn.ModuleList([PromptGeneratorLayer(embed_dim, embed_dim//64) for _ in range(layers)])
        self.alpha = nn.Parameter(torch.ones(embed_dim) * alpha)
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    
    def forward(self, text, visual):
        B, N, C = visual.shape
        visual = self.norm(visual)
        for layer in self.decoder:
            text = layer(text, visual)
        
        return self.alpha * text

class PromptPool(nn.Module):
    def __init__(self, pool_size, embedd_dim, use_freq=False, pool_prompts_per_sample=5, pool_prompt_length=5) -> None:
        super().__init__()
        self.pool_size = pool_size
        self.embedd_dim = embedd_dim
        self.use_freq = use_freq
        self.pool_prompts_per_sample = pool_prompts_per_sample
        self.keys = nn.Parameter(torch.empty([pool_size, embedd_dim]).uniform_(0, 0.01))
        self.values = nn.Parameter(torch.empty([pool_size, pool_prompt_length, embedd_dim]).uniform_(0, 0.01))
        # self.prompt_freq = torch.ones([pool_size]).requires_grad_(False)
        self.register_buffer("prompt_freq", torch.ones([pool_size]).requires_grad_(False))
        # torch.autograd.set_detect_anomaly(True)
    
    def forward(self, x):
        # x.shape = [b*t, 1, d]: shape of the [class token]
        key_loss = None
        self.prompt_freq = self.prompt_freq.to(self.keys.device)
        # with torch.no_grad():
        #     self.keys.data = self.keys.data/self.keys.data.norm(dim=-1).unsqueeze(dim=-1)
        #     # self.values.data = self.values.data/self.values.data.norm(dim=-1).unsqueeze(dim=-1)
        #     x.data = x.data/x.data.norm(dim=-1).unsqueeze(dim=-1)

        if self.use_freq and self.training:
            penalty = 1 - self.prompt_freq/self.prompt_freq.sum() #[1, pool_size]
            penalty = penalty/penalty.sum()
            
            cosine_distance = 1 - penalty.clone()*torch.cosine_similarity(x, self.keys, dim=-1).reshape(x.shape[0], self.pool_size)        
        else:
            cosine_distance = 1 - torch.cosine_similarity(x, self.keys, dim=-1).reshape(x.shape[0], self.pool_size)        

        cosine_distance, idx = cosine_distance.topk(self.pool_prompts_per_sample, dim=-1, largest=False) # [x.shape[0], k]

        if self.training: # if in train mode
            if self.use_freq:
                selected_prompts, freqs = torch.unique(idx, return_counts=True)
                selected_prompts = selected_prompts.to(self.prompt_freq.device)
                freqs = freqs.to(self.prompt_freq.device)
                
                # penalty = 1 - self.prompt_freq/self.prompt_freq.sum() #[1, pool_size]
                # penalty = penalty/penalty.sum()
                # key_loss = cosine_distance * penalty[idx]

                for i, prompt in enumerate(selected_prompts):
                    self.prompt_freq[prompt] += freqs[i]
            key_loss = cosine_distance.mean()

        return self.values[idx, :].reshape(x.shape[0], -1, self.embedd_dim), key_loss
