from collections import OrderedDict
from timm.models.layers import trunc_normal_
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint_sequential
import sys
sys.path.append("../")
from clip.model import LayerNorm, QuickGELU, DropPath
from .prompt import PromptPool


class CrossFramelAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, droppath = 0., T=0, ):
        """ 
            d_model: embedding dim of the message token
            T: time dimension -> number of frames extracted from a video
        """
        super().__init__()
        self.T = T

        self.message_fc = nn.Linear(d_model, d_model)
        self.message_ln = LayerNorm(d_model)
        self.message_attn = nn.MultiheadAttention(d_model, n_head,)
           
        self.attn = nn.MultiheadAttention(d_model, n_head,)
        self.ln_1 = LayerNorm(d_model)
        
        self.drop_path = DropPath(droppath) if droppath > 0. else nn.Identity()
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]


    def forward(self, x):
        l, bt, d = x.size()
        b = bt // self.T
        x = x.view(l, b, self.T, d) 

        msg_token = self.message_fc(x[0,:,:,:]) 
        msg_token = msg_token.view(b, self.T, 1, d) 
        
        msg_token = msg_token.permute(1,2,0,3).view(self.T, b, d) 
        msg_token = msg_token + self.drop_path(self.message_attn(self.message_ln(msg_token),self.message_ln(msg_token),self.message_ln(msg_token),need_weights=False)[0])
        msg_token = msg_token.view(self.T, 1, b, d).permute(1,2,0,3) # -> x.shape = [1, b, T, d]
        
        x = torch.cat([x, msg_token], dim=0)
        
        x = x.view(l+1, -1, d) # x.shape = [l+1, bt, d]
        x = x + self.drop_path(self.attention(self.ln_1(x)))
        # The message token is only used for attention across frame and is dropped after fedding into the last FFN(self.mlp)
        x = x[:l,:,:]
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, droppath=None, use_checkpoint=False, T=8):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        if droppath is None:
            droppath = [0.0 for i in range(layers)] 
        self.width = width
        self.layers = layers
        
        self.resblocks = nn.Sequential(*[CrossFramelAttentionBlock(width, heads, attn_mask, droppath[i], T) for i in range(layers)])
       
    def forward(self, x: torch.Tensor):
        if not self.use_checkpoint:
            return self.resblocks(x)
        else:
            return checkpoint_sequential(self.resblocks, 3, x)


class CrossFrameCommunicationTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int,
                 droppath = None, T = 8, use_checkpoint = False, 
                 pool_size:int = 0, pool_use_freq=False, pool_prompts_per_sample=5, pool_prompt_length=5):
        """
            width: size of the message token as well as class embedding
            class embedding is a token prepened at the beginning of the sequence of image patch
        """
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.pool_size = pool_size
        self.pool_prompt_length = pool_prompt_length
        self.pool_prompt_per_sample = pool_prompts_per_sample
        # to extract tokens out of the original image
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.prompt_pool = None
        ## Prompt pool
        if pool_size > 0:
            self.prompt_pool = PromptPool(pool_size=pool_size, 
                                          embedd_dim=width, 
                                          use_freq=pool_use_freq, 
                                          pool_prompts_per_sample=pool_prompts_per_sample,
                                          pool_prompt_length=pool_prompt_length)
        
        # (input_resolution // patch_size) ** 2 + 1 = number of token (grid **2) + 1 class token 
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        ## Attention Blocks
        self.transformer = Transformer(width, layers, heads, droppath=droppath, use_checkpoint=use_checkpoint, T=T,)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor):
        # x.shape = b*t*h*w= batchsize * num_frames extracted from a video
        x = self.conv1(x)  # shape = [*, width, grid, grid] grid: number of token along 1 spatial dimension
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2] -> flatten to a matrix of tokens
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width] -> each token has embedding dim = width
        # prepending the class_embedding token to the begining of the sequence
        cls = self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls, x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        #@TODO: use the [class] token as query to query the prompt pool
        prompt_key_loss = None
        if self.prompt_pool is not None:
            query = x.clone().detach()
            with torch.no_grad():
                query = self.ln_pre(query)
                query = query.permute(1, 0, 2)
                query = self.transformer(query)
                query = query.permute(1, 0, 2)
                query = self.ln_post(query[:, 0, :])
                query = query.unsqueeze(1)

            prompt, prompt_key_loss = self.prompt_pool(query)
            # prompt_key_loss = None
            x = torch.cat([prompt, x], dim=1)
        # print("x.requires_grad", x.requires_gra)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2) # [b*t, grid**2 + 1(class token), width]-> [grid**2 + 1 (class token), (b*t), width] this is needed for multihead self attn when batch_first = false
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        if self.pool_size > 0:
            prompt_idx = self.pool_prompt_per_sample*self.pool_prompt_length
        else:
            prompt_idx = 0 

        cls_x = self.ln_post(x[:, 0:prompt_idx+1, :].mean(dim=1))
        if self.proj is not None:
            cls_x = cls_x @ self.proj
        return cls_x, x[:,prompt_idx+1:,:], prompt_key_loss # you have to skip the [cls] token as input to the prompt generato