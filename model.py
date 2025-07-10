import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
from torch import Tensor

class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40, in_channels=1):
        super().__init__()
        # 投影层
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=(1, 50), stride=(1, 25)),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2),
            nn.Conv2d(8, emb_size, kernel_size=(1, 5), stride=(1, 1)),
            nn.BatchNorm2d(emb_size),
            nn.LeakyReLU(0.2),
            Rearrange('b e c t -> b (c t) e'),
        )
        
        # 延迟初始化位置编码
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.position_embedding = None
        self.emb_size = emb_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        
        # 动态创建位置编码（如果尚未创建）
        if self.position_embedding is None:
            num_patches = x.shape[1]
            # 确保位置编码在正确的设备上
            device = x.device
            self.position_embedding = nn.Parameter(
                torch.randn(1, num_patches + 1, self.emb_size, device=device)
            )
            # print(f"动态创建位置编码: (1, {num_patches+1}, {self.emb_size}) on {device}")
        
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.position_embedding
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads=4, dropout=0.1):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x, mask=None):
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = torch.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return x + self.fn(x, **kwargs)

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion=4, drop_p=0.1):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
            nn.Dropout(drop_p)
        )

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size, num_heads=4, drop_p=0.1, forward_expansion=4):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(emb_size, forward_expansion, drop_p),
                nn.Dropout(drop_p)
            ))
        )

class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])

class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes=2):
        super().__init__(
            nn.LayerNorm(emb_size),
            Reduce('b n e -> b e', reduction='mean'),
            nn.Linear(emb_size, n_classes)
        )

class EEGTransformer(nn.Module):
    def __init__(self, emb_size=40, depth=4, n_classes=2):
        super().__init__()
        self.patch_embedding = PatchEmbedding(emb_size)
        self.transformer = TransformerEncoder(depth, emb_size)
        self.classifier = ClassificationHead(emb_size, n_classes)
        
    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.transformer(x)
        x = self.classifier(x)
        return x