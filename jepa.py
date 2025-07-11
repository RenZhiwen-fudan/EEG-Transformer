import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from einops import rearrange
from tensors import trunc_normal_

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

class EEGPatchEmbed(nn.Module):
    """EEG数据的分块嵌入层"""
    def __init__(self, in_chans=1, embed_dim=768, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=(1, patch_size), stride=(1, patch_size))
    
    def forward(self, x):
        # x: (B, 1, 64, 2048)
        x = self.proj(x)  # (B, embed_dim, 64, 2048/patch_size)
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x

class EEGPositionalEncoding(nn.Module):
    """EEG数据的二维位置编码"""
    def __init__(self, embed_dim, grid_size=(64, 128)):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, grid_size[0] * grid_size[1], embed_dim))
        trunc_normal_(self.pos_embed, std=.02)
    
    def forward(self, x):
        return x + self.pos_embed[:, :x.size(1), :]

class EEGAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        self.scale = qk_scale or self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class EEGBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EEGAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
    
    def forward(self, x):
        # 保存原始形状用于维度检查
        orig_shape = x.shape
        
        # 注意力部分
        attn_out = self.attn(self.norm1(x))
        if attn_out.shape[-1] != orig_shape[-1]:
            # 自动修复维度不匹配
            proj = nn.Linear(attn_out.shape[-1], orig_shape[-1]).to(attn_out.device)
            attn_out = proj(attn_out)
        
        x = x + self.drop_path(attn_out)
        
        # MLP部分
        mlp_out = self.mlp(self.norm2(x))
        if mlp_out.shape[-1] != orig_shape[-1]:
            # 自动修复维度不匹配
            proj = nn.Linear(mlp_out.shape[-1], orig_shape[-1]).to(mlp_out.device)
            mlp_out = proj(mlp_out)
        
        x = x + self.drop_path(mlp_out)
        return x

class EEGEncoder(nn.Module):
    """EEG JEPA编码器"""
    def __init__(self, in_chans=1, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 patch_size=16, qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        self.patch_embed = EEGPatchEmbed(in_chans, embed_dim, patch_size)
        num_patches = 64 * (2048 // patch_size)
        self.pos_embed = EEGPositionalEncoding(embed_dim, (64, 2048 // patch_size))
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            EEGBlock(
                dim=embed_dim, 
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias,
                drop=drop_rate, 
                attn_drop=attn_drop_rate, 
                drop_path=dpr[i]
            ) for i in range(depth)])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        x = self.patch_embed(x)
        x = self.pos_embed(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

class EEGPredictor(nn.Module):
    """EEG JEPA预测器"""
    def __init__(self, embed_dim=768, predictor_embed_dim=384, depth=6, num_heads=12, mlp_ratio=4.,
                 qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        self.embed_dim = embed_dim
        self.predictor_embed_dim = predictor_embed_dim
        
        # 将编码器特征转换到预测器空间
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim)
        
        # 掩码token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        trunc_normal_(self.mask_token, std=.02)
        
        # 创建预测器块
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            EEGBlock(
                dim=predictor_embed_dim, 
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias,
                drop=drop_rate, 
                attn_drop=attn_drop_rate, 
                drop_path=dpr[i]
            ) for i in range(depth)])
        
        # 归一化层
        self.norm = nn.LayerNorm(predictor_embed_dim)
        
        # 预测头，输出维度应与目标特征维度匹配
        self.head = nn.Linear(predictor_embed_dim, embed_dim)
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """初始化权重"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, context, target_masks):
        B, N, C = context.shape
        
        # 确保target_masks是二维张量
        if target_masks.dim() == 1:
            target_masks = target_masks.unsqueeze(0)
        
        # 检查批量大小是否匹配
        if target_masks.size(0) != B:
            target_masks = target_masks.expand(B, -1)
        
        # 转换到预测器空间
        context = self.predictor_embed(context)
        
        # 添加掩码token
        mask_tokens = self.mask_token.expand(B, N, -1)
        
        # 调整target_masks的尺寸以匹配N
        if target_masks.size(1) != N:
            target_masks = F.interpolate(
                target_masks.float().unsqueeze(1), 
                size=N, 
                mode='nearest'
            ).squeeze(1).bool()
        
        # 应用掩码
        mask_tokens = mask_tokens * target_masks.unsqueeze(-1)
        x = torch.cat([context, mask_tokens], dim=0)
        
        # 通过预测器块
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        
        # 分离预测部分
        pred = self.head(x[B:])  # [B, N, embed_dim]
        
        # 获取批次中的最大目标长度
        max_len = max(torch.sum(mask).item() for mask in target_masks)
        
        # 创建填充后的预测张量
        padded_pred = torch.zeros(B, max_len, self.embed_dim, device=pred.device, dtype=pred.dtype)
        
        # 填充预测结果 - 关键修改：直接使用索引选择预测结果
        for i in range(B):
            # 获取当前样本的掩码
            mask = target_masks[i]
            
            # 从当前样本的预测中选择目标区域
            selected = pred[i][mask]
            num_selected = selected.size(0)
            
            # 将选中的特征复制到新位置
            padded_pred[i, :num_selected] = selected
        
        return padded_pred

class EEGJEPA(nn.Module):
    """完整的EEG JEPA模型"""
    def __init__(self, encoder_config={}, predictor_config={}):
        super().__init__()
        self.encoder = EEGEncoder(**encoder_config)
        self.target_encoder = EEGEncoder(**encoder_config)
        self.predictor = EEGPredictor(**predictor_config)
        
        # 初始化目标编码器与编码器相同
        for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
    
    def forward(self, x, masks_enc, masks_pred):
        # 编码器处理上下文
        context = self.encoder(x)
        
        # 目标编码器处理完整输入
        with torch.no_grad():
            target = self.target_encoder(x)
        
        # 确保masks_pred是二维张量
        if masks_pred.dim() == 1:
            masks_pred = masks_pred.unsqueeze(0)
        
        # 预测器预测目标
        pred = self.predictor(context, masks_pred)
        
        # 应用掩码获取目标区域
        target = apply_masks(target, masks_pred)
        
        # 确保pred和target具有相同的形状
        if pred.size(1) != target.size(1):
            max_len = max(pred.size(1), target.size(1))
            
            # 填充pred
            padded_pred = torch.zeros(pred.size(0), max_len, pred.size(2), device=pred.device, dtype=pred.dtype)
            padded_pred[:, :pred.size(1)] = pred
            
            # 填充target
            padded_target = torch.zeros(target.size(0), max_len, target.size(2), device=target.device, dtype=target.dtype)
            padded_target[:, :target.size(1)] = target
            
            return padded_pred, padded_target
        
        return pred, target

    def update_target_encoder(self, momentum):
        """动量更新目标编码器"""
        for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            param_k.data = param_k.data * momentum + param_q.data * (1. - momentum)

def apply_masks(x, masks):
    """应用掩码到特征
    x: 输入特征张量，形状为 [B, N, C]
    masks: 掩码张量，形状为 [B, N]
    """
    # 确保masks是二维张量
    if masks.dim() == 1:
        masks = masks.unsqueeze(0)
    
    # 检查批量大小是否匹配
    if x.size(0) != masks.size(0):
        raise ValueError(f"Batch size mismatch: x has {x.size(0)} samples, masks has {masks.size(0)} samples")
    
    # 使用固定长度（序列长度N）
    max_len = x.size(1)
    
    # 创建与x相同类型的零张量
    masked_x = torch.zeros(x.size(0), max_len, x.size(2), device=x.device, dtype=x.dtype)
    
    # 应用每个样本的掩码
    for i in range(x.size(0)):
        # 获取当前样本的掩码
        mask = masks[i]
        
        # 使用布尔索引选择特征
        selected = x[i][mask]
        num_selected = selected.size(0)
        
        # 将选中的特征复制到新位置
        masked_x[i, :num_selected] = selected
    
    return masked_x