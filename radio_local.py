import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Im2Patches(nn.Module):
    def __init__(self, patch_size=16):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x):
        B, C, H, W = x.shape
        P = self.patch_size
        x = x.unfold(2, P, P).unfold(3, P, P)
        x = x.contiguous().view(B, C, H // P, W // P, P, P)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous().view(B, -1, C * P * P)
        return x

class ViTPatchLinear(nn.Linear):
    def __init__(self, in_features=768, out_features=1024, bias=False):
        super().__init__(in_features, out_features, bias=bias)

class ClsToken(nn.Module):
    def __init__(self, ndim: int, num_tokens: int = 8, enabled: bool = True):
        super().__init__()
        self.ndim = ndim
        self.enabled = enabled
        self.num_tokens = num_tokens

        if enabled:
            scale = ndim ** -0.5
            self.token = nn.Parameter(torch.randn(num_tokens, ndim) * scale)
        else:
            self.token = None

    def forward(self, x: torch.Tensor):
        if self.token is None:
            return x
        token = self.token.unsqueeze(0).expand(x.shape[0], -1, -1)
        return torch.cat([token, x], dim=1)

class ViTPatchGenerator(nn.Module):
    def __init__(self, patch_size=16, in_chans=3, embed_dim=1024, num_prefix_tokens=8):
        super().__init__()
        self.im_to_patches = Im2Patches(patch_size)
        self.embedder = ViTPatchLinear(in_features=in_chans * patch_size * patch_size, out_features=embed_dim, bias=False)
        self.cls_token = ClsToken(ndim=embed_dim, num_tokens=num_prefix_tokens)
        self.patch_normalizer = nn.Identity()
        
        self.pos_embed = nn.Parameter(torch.zeros(1, 16384, embed_dim))

    def forward(self, x):
        x = self.im_to_patches(x)
        x = self.embedder(x)
        x = self.cls_token(x)
        x = self.patch_normalizer(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer(approximate='none')
        self.drop1 = nn.Dropout(drop)
        self.norm = nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=16, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.norm = nn.Identity()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = self.q_norm(q)
        k = self.k_norm(k)

        x = F.scaled_dot_product_attention(
            q, k, v, 
            dropout_p=self.attn_drop.p if self.training else 0.0
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-06)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.ls1 = nn.Identity()
        self.drop_path1 = nn.Identity()
        
        self.norm2 = nn.LayerNorm(dim, eps=1e-06)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), out_features=dim)
        self.ls2 = nn.Identity()
        self.drop_path2 = nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4.0, num_prefix_tokens=8):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_prefix_tokens = num_prefix_tokens
        
        self.patch_embed = None
        self.pos_drop = None
        self.patch_drop = nn.Identity()
        self.norm_pre = nn.Identity()
        
        self.patch_generator = ViTPatchGenerator(
            patch_size=patch_size, 
            embed_dim=embed_dim, 
            num_prefix_tokens=num_prefix_tokens
        )
        
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        
        self.norm = nn.Identity()
        self.fc_norm = nn.Identity()
        self.head_drop = nn.Dropout(0.0)
        self.head = nn.Identity()

    def _interpolate_pos_encoding(self, x, H, W):
        pos_embed = self.patch_generator.pos_embed 
        L = pos_embed.shape[1]
        
        if int(math.sqrt(L - self.num_prefix_tokens))**2 == L - self.num_prefix_tokens:
            pos_prefix_len = self.num_prefix_tokens
        else:
            base_grid = int(math.sqrt(L))
            pos_prefix_len = L - (base_grid ** 2)
            
        base_spatial_len = L - pos_prefix_len
        base_grid = int(math.sqrt(base_spatial_len))
        
        extra_pos_embed = pos_embed[:, :pos_prefix_len]
        patch_pos_embed = pos_embed[:, pos_prefix_len:]
        
        dim = x.shape[-1]
        w0, h0 = W // self.patch_size, H // self.patch_size
        
        if (h0 != base_grid or w0 != base_grid):
            patch_pos_embed = patch_pos_embed.reshape(1, base_grid, base_grid, dim).permute(0, 3, 1, 2)
            patch_pos_embed = F.interpolate(patch_pos_embed, size=(h0, w0), mode='bicubic', align_corners=False)
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
            
        if pos_prefix_len < self.num_prefix_tokens:
            padding = torch.zeros(1, self.num_prefix_tokens - pos_prefix_len, dim, 
                                  device=pos_embed.device, dtype=pos_embed.dtype)
            extra_pos_embed = torch.cat((extra_pos_embed, padding), dim=1)
            
        elif pos_prefix_len > self.num_prefix_tokens:
            extra_pos_embed = extra_pos_embed[:, :self.num_prefix_tokens]
            
        return torch.cat((extra_pos_embed, patch_pos_embed), dim=1)

    def forward(self, x, original_H, original_W):
        x = self.patch_generator(x)
        x = self.norm_pre(x)
        x = x + self._interpolate_pos_encoding(x, original_H, original_W)
        x = self.patch_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = self.fc_norm(x)
        x = self.head_drop(x)
        x = self.head(x)
        return x

class InputConditioner(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("norm_mean", torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1))
        self.register_buffer("norm_std", torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1))

    def forward(self, x):
        return (x - self.norm_mean) / self.norm_std

class RADIOModel(nn.Module):
    def __init__(self, num_prefix_tokens=8): 
        super().__init__()
        self.model = VisionTransformer(num_prefix_tokens=num_prefix_tokens)
        self.input_conditioner = InputConditioner()
        self.adaptors = nn.ModuleDict()
        self.feature_normalizer = nn.Identity()
        
        self.register_buffer("summary_idxs", torch.tensor([0, 1, 2])) 

    def forward(self, x, feature_fmt='NCHW'):
        B, C, H, W = x.shape
        x = self.input_conditioner(x)
        x = self.model(x, original_H=H, original_W=W)
        x = self.feature_normalizer(x)
        
        summary = x[:, self.summary_idxs] 
        spatial_features = x[:, self.model.num_prefix_tokens:]
        
        B, C, D = summary.shape
        summary = summary.reshape(B, C*D)
        return summary, spatial_features