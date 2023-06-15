from typing import Tuple, Union
import torch
from torch import nn
import numpy as np
from .mit import MultiframeIntegrationTransformer
from .prompt import VideoSpecificPrompt, PromptPool
from .cct import CrossFrameCommunicationTransformer
import sys
import warnings
sys.path.append("../")
from clip.model import CLIP,LayerNorm,Transformer
import clip

class XCLIP(CLIP):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int, 
                 # video
                 T=8, 
                 droppath=0.,
                 mit_layers=1,
                 # prompt 
                 prompts_alpha=1e-4,
                 prompts_layers=1,
                 # other
                 use_cache=True,
                 use_checkpoint=False,
                 pool_size=25, # number of prompts in the prompt pool
                 pool_use_freq=False,
                 pool_prompts_per_sample=5,
                 pool_prompt_length=5,
                 pool_freeze_video=False,
                 num_classes = 14,
                 context_prompt_len=8,
                 class_prompt_len=8,
                 fine_grain_loss=False
                 ):
        super().__init__(
            embed_dim,
            image_resolution, vision_layers, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
        )
        self.pool_size = pool_size
        self.pool_use_freq = pool_use_freq
        self.pool_prompts_per_sample = pool_prompts_per_sample
        self.pool_prompt_length = pool_prompt_length
        self.context_prompt_len = context_prompt_len
        self.class_prompt_len = class_prompt_len
        self.fine_grain_loss = fine_grain_loss
        
        if not use_cache:
            if context_prompt_len > 0:
                self.prompt_context_prefix = nn.Parameter(torch.empty(context_prompt_len, transformer_width).normal_(mean=0, std=0.02))
                self.prompt_context_postfix = nn.Parameter(torch.empty(context_prompt_len, transformer_width).normal_(mean=0, std=0.02))
            if class_prompt_len > 0:
                self.prompt_class_prefix = nn.Parameter(torch.empty(num_classes, class_prompt_len, transformer_width).normal_(mean=0, std=0.02))
                self.prompt_class_postfix = nn.Parameter(torch.empty(num_classes, class_prompt_len, transformer_width).normal_(mean=0, std=0.02))
        self.transformer_width = transformer_width # = text embedding dim

        self.prompts_generator = VideoSpecificPrompt(layers=prompts_layers, embed_dim=embed_dim, alpha=prompts_alpha,)
        self.use_cache=use_cache
        self.mit = MultiframeIntegrationTransformer(T=T, embed_dim=embed_dim, layers=mit_layers,)

        dpr = [x.item() for x in torch.linspace(0, droppath, vision_layers)] if droppath > 0. else None

        vision_heads = vision_width // 64
        self.visual = CrossFrameCommunicationTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim,
            droppath=dpr,
            T=T,
            use_checkpoint=use_checkpoint,
            pool_size=pool_size, 
            pool_use_freq=pool_use_freq,
            pool_prompts_per_sample=pool_prompts_per_sample,
            pool_prompt_length=pool_prompt_length
        )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.cache_text_features = None
        self.prompts_visual_ln = LayerNorm(vision_width)
        self.prompts_visual_proj = nn.Parameter(torch.randn(vision_width, embed_dim))
        

        self.initialize_parameters()
        # if training with prompt pool -> can choose whether or not to freeze the video encoder
        if pool_freeze_video:
            self.freeze_no_prompt()
        
        if self.fine_grain_loss:
            # for coarse-grained constrast weights
            self.global_mat_weight = nn.parameter.Parameter(torch.eye(embed_dim), requires_grad=True)

            # for cross-grained constrast weights
            num_words = 77
            self.word_logit_weight = nn.parameter.Parameter(torch.eye(num_words), requires_grad=True)
            self.frame_logit_weight = nn.parameter.Parameter(torch.eye(T), requires_grad=True)        

            # for fine-grained constrast weights
            self.local_mat_weight = nn.parameter.Parameter(torch.eye(embed_dim), requires_grad=True)
            self.frame_mat_weight = nn.parameter.Parameter(torch.eye(T), requires_grad=True) # T = num_frames
            self.word_mat_weight = nn.parameter.Parameter(torch.eye(num_words), requires_grad=True)
            self.frame_mat_weight2 = nn.parameter.Parameter(torch.eye(T), requires_grad=True)
            self.word_mat_weight2 = nn.parameter.Parameter(torch.eye(num_words), requires_grad=True)

        if self.use_cache:
            self.freeze_module(self.transformer)
            self.freeze_module(self.token_embedding)
            self.freeze_module(self.ln_final)
            self.text_projection.requires_grad_(False)
            self.positional_embedding.requires_grad_(False)
    

    def _attenion_over_fine_grained_sim_matrix(self, word_features, frame_features):
        bs_video, num_frames, dim_video = frame_features.shape
        bs_text, num_words, dim_text = word_features.shape
        fine_grained_sim_scores = torch.matmul(torch.matmul(word_features.view(-1, dim_text), self.local_mat_weight), frame_features.view(-1, dim_video).t()).view(bs_text, num_words, bs_video, num_frames)  # [bs_text, num_words, bs_video, num_frames]

        word_level_logit = torch.sum(torch.matmul(torch.softmax(fine_grained_sim_scores/1e-2, dim=1).permute(0,2,3,1), self.word_mat_weight).permute(0,3,1,2) * fine_grained_sim_scores, dim=1)               # [bs_text, bs_video, num_frames]
        frame_level_logit = torch.sum(torch.matmul(torch.softmax(fine_grained_sim_scores/1e-2, dim=-1), self.frame_mat_weight) * fine_grained_sim_scores, dim=-1)                                             # [bs_text, num_words, bs_video]

        sent2frame_logits = torch.sum(torch.matmul(torch.softmax(word_level_logit/1e-2, dim=-1),self.frame_mat_weight2) * word_level_logit, dim=-1)                                # [bs_text, bs_video]
        video2word_logits = torch.sum(torch.matmul(torch.softmax(frame_level_logit/1e-2, dim=1).permute(0,2,1), self.word_mat_weight2).permute(0,2,1) * frame_level_logit, dim=1)  # [bs_text, bs_video]

        return (sent2frame_logits + video2word_logits) / 2

    def freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad_(False)  
    
    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'positional_embedding'}

    def encode_image(self, image):
        return self.visual(image)

    def prompt_text(self, x:torch.Tensor, text_mask:torch.Tensor):
        """
            Given the tokenized + embedded string -> replace some padding token with learnable prompt
            x: a text vector after the embedding layer. x.shape = [num_text, ctx_len=77, embedding_dim]
            text_mask: a binary mask which mask out padding element
        """
        context_prompt_len = self.context_prompt_len
        class_prompt_len = self.class_prompt_len
        eos_position = []

        mask_token = self.token_embedding(torch.IntTensor([0]).to(x.device)) # index 0 is the mask token

        prompted_text = torch.zeros([x.shape[0], 77, self.transformer_width]).to(x.device)
        prompted_text[:, 0, :] = self.token_embedding(torch.IntTensor([49406]).to(x.device)) # start of sentence embedding
        eos = self.token_embedding(torch.IntTensor([49407]).to(x.device)) # end of sentence embedding
        
        if context_prompt_len > 0:
            prompted_text[:, 1:context_prompt_len+1, :] = self.prompt_context_prefix

        for idx, category in enumerate(x):
            category_len = text_mask[idx].sum()-1 # number of text token in a category except for the start token
            if class_prompt_len > 0:
                prompted_text[idx, context_prompt_len+1:context_prompt_len+class_prompt_len+1, :] = self.prompt_class_prefix[idx] # class prefix
            prompted_text[idx, context_prompt_len+class_prompt_len+1: context_prompt_len+class_prompt_len+1 + category_len-1, :] = category[text_mask[idx]][1:-1]
            if class_prompt_len > 0:
                prompted_text[idx, context_prompt_len+class_prompt_len+1 + category_len-1:context_prompt_len+class_prompt_len*2+1 + category_len -1, :] = self.prompt_class_postfix[idx] # class prefix
            if context_prompt_len > 0:
                prompted_text[idx, context_prompt_len+class_prompt_len*2+1 + category_len -1:context_prompt_len*2+class_prompt_len*2+1 + category_len-1, : ] = self.prompt_context_postfix

            prompted_text[idx, context_prompt_len*2+class_prompt_len*2+1 + category_len-1, :] = eos
            
            eos_position.append(context_prompt_len*2+class_prompt_len*2+1 + category_len-1)

            prompted_text[idx, context_prompt_len*2+class_prompt_len*2 + category_len+1:, :] = mask_token.repeat((77-context_prompt_len*2 - class_prompt_len*2-category_len-1, 1))
            
        return prompted_text, eos_position

    def encode_text(self, text):
        x = self.token_embedding(text)
        eos_indx = text.argmax(dim=-1)
        K, N1, C = x.shape

        if not self.use_cache:
            x, eos_indx = self.prompt_text(x, text_mask=text!=0)

        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        if self.fine_grain_loss:
            word_features = x @ self.text_projection
        x = x[torch.arange(x.shape[0]), eos_indx] @ self.text_projection
        x = x.reshape(K, -1)

        if self.fine_grain_loss:
            return x, word_features
        return x, None

    def encode_video(self, image):
        b,t,c,h,w = image.size()
        image = image.reshape(-1,c,h,w) #[bach_size*num_frame, c, h, w]

        cls_features, img_features, prompt_key_loss = self.encode_image(image)
        
        img_features = self.prompts_visual_ln(img_features)
        img_features = img_features @ self.prompts_visual_proj
        
        cls_features = cls_features.view(b, t, -1) # [b*t, 1, width] -> [b, t, width]

        # if self.pool_size > 0:
        #     cls_features = cls_features.view(b*t, 1, -1)
        #     cls_features, prompt_key_loss_2 = self.prompt_pool(cls_features)
        #     cls_features = cls_features.mean(dim=1)
        #     if self.training:
        #         prompt_key_loss += prompt_key_loss_2
        #     cls_features = cls_features.view(b, t, -1)

        img_features = img_features.view(b,t,-1,cls_features.shape[-1])
        #@TODO: prompt goes here
        video_features = self.mit(cls_features)

        return video_features, img_features, prompt_key_loss

    def cache_text(self, text):
        """This function is used when freezing the text encoder during training"""
        self.eval()
        with torch.no_grad():
            if self.cache_text_features is None:
                self.cache_text_features = self.encode_text(text)
        self.train()
        return self.cache_text_features
    
    def freeze_no_prompt(self): 
        for name, param in self.named_parameters():
            if 'visual.prompt_pool' in name or "prompt_context_postfix" in name or "prompt_context_prefix" in name \
                or "prompt_class_prefix" in name or "prompt_class_postfix" in name or "logit_scale" in name:
                print("unfreeze", name)
                param.requires_grad_(True)
                # or "mit." in name or "visual.class_embedding" in name or "prompts_generator" in name or "prompt_context_prefix" in name or "prompt_context_postfix" in name
            else:
                param.requires_grad_(False)


    def forward(self, image, text):
        b = image.shape[0]
        frame_features, img_features, prompt_key_loss = self.encode_video(image) 
        img_features = img_features.mean(dim=1, keepdim=False)

        if self.use_cache:
            text_features, _ = self.cache_text(text)
        else:
            text_features, word_features = self.encode_text(text)
        
        
        text_features = text_features.unsqueeze(0).expand(b, -1, -1)
        text_features = text_features + self.prompts_generator(text_features, img_features)
        
        video_features = frame_features.mean(dim=1, keepdim=False)

        video_features = video_features / video_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        if self.fine_grain_loss:
            word_features = word_features / word_features.norm(dim=-1, keepdim=True)
            frame_features = frame_features / frame_features.norm(dim=-1, keepdim=True)
            # video-sentence score 
            video_sentence_logits = logit_scale * torch.einsum("bkd, bd->kb", torch.matmul(text_features, self.global_mat_weight), video_features)

            # video-word score
            video_word_logits = logit_scale * torch.sum(torch.matmul(word_features, video_features.t()) \
                * torch.matmul(torch.softmax(torch.matmul(word_features, video_features.t()) / 1e-2, dim=1).permute(0,2,1), self.word_logit_weight).permute(0,2,1), dim=1)

            # sentence-frame score 
            sentence_frame_logits = logit_scale * torch.sum(torch.matmul(text_features, frame_features.permute(0, 2, 1)) \
                * torch.matmul(torch.softmax(torch.matmul(text_features, frame_features.permute(0, 2, 1)) / 1e-2, dim=-1), self.frame_logit_weight), dim=-1).t()

            # frame-word score
            frame_word_logits = logit_scale * self._attenion_over_fine_grained_sim_matrix(word_features, frame_features)

            logits = (video_sentence_logits + video_word_logits + sentence_frame_logits + frame_word_logits) / 4
            logits = logits.permute(1, 0)
        else: 
           logits = torch.einsum("bd,bkd->bk", video_features, logit_scale * text_features)
        
        return logits, prompt_key_loss


def build_model(state_dict: dict, 
                T=8, 
                droppath=0., 
                use_checkpoint=False, 
                logger=None, 
                prompts_alpha=1e-1, 
                prompts_layers=2, 
                use_cache=True, 
                mit_layers=4, 
                pool_size=25, 
                pool_use_freq=False, # use frequency counting for prompt layer
                pool_prompts_per_sample=5,
                pool_prompt_length=5,
                pool_freeze_video=False,
                num_classes=14,
                context_prompt_len=8,
                class_prompt_len=8,
                fine_grain_loss=False
                ):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))
    
    model = XCLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,  
        T=T, droppath=droppath, mit_layers=mit_layers,
        prompts_alpha=prompts_alpha, prompts_layers=prompts_layers,
        use_checkpoint=use_checkpoint, use_cache=use_cache,
        pool_size=pool_size,
        pool_use_freq=pool_use_freq,
        pool_prompts_per_sample=pool_prompts_per_sample,
        pool_prompt_length=pool_prompt_length,
        pool_freeze_video=pool_freeze_video,
        num_classes=num_classes,
        context_prompt_len=context_prompt_len,
        class_prompt_len=class_prompt_len,
        fine_grain_loss=fine_grain_loss
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    msg = model.load_state_dict(state_dict,strict=False)
    logger.info(f"load pretrained CLIP: {msg}")
    
    return model.eval()


def load(model_path, name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", 
         jit=True, T=8, droppath=0., use_checkpoint=False, logger=None, use_cache=True, prompts_alpha=1e-1, prompts_layers=2, mit_layers=1,
         pool_size=25,
         pool_use_freq=False,
         pool_prompts_per_sample=5,
         pool_prompt_length=5,
         pool_freeze_video=False,
         num_classes=14,
         context_prompt_len=8,
         class_prompt_len=8,
         fine_grain_loss = False
):
    if model_path is None:
        model_path = clip._download(clip._MODELS[name])
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location=device if jit else "cpu").eval()
        print("it's jit pls")
        state_dict = None
    except RuntimeError:
        # loading saved state dict
        if jit:
            warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
            jit = False
        print("helooooo not jit pls")
        state_dict = torch.load(model_path, map_location="cpu")

    model = build_model(state_dict or model.state_dict(), T=T, droppath=droppath, 
                        use_checkpoint=use_checkpoint, logger=logger,
                        prompts_alpha=prompts_alpha, 
                        prompts_layers=prompts_layers,
                        use_cache=use_cache,
                        mit_layers=mit_layers,
                        pool_size=pool_size,
                        pool_use_freq=pool_use_freq,
                        pool_prompts_per_sample=pool_prompts_per_sample,
                        pool_prompt_length=pool_prompt_length,
                        pool_freeze_video=pool_freeze_video,
                        num_classes=num_classes,
                        context_prompt_len=context_prompt_len,
                        class_prompt_len=class_prompt_len,
                        fine_grain_loss=fine_grain_loss
                        )
    if str(device) == "cpu":
        model.float()
    return model.cuda()