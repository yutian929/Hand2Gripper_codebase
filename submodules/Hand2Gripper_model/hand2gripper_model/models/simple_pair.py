# model.py
# -*- coding: utf-8 -*-
"""
Hand-to-Gripper (2-finger) Mapping Model (Left/Right Only)

This model maps hand keypoints to the left and right fingertips of the gripper, no longer predicting the base point.

Input:
- color:        [B, 3, H, W]    RGB image
- bbox:         [B, 4]          hand bounding box [x1,y1,x2,y2]
- keypoints_3d: [B, 21, 3]      hand 3D keypoints
- contact:      [B, 21]         contact probability (parameter retained but not used)
- is_right:     [B]             1=right hand, 0=left hand (as feature input)

Output:
- logits_left:  [B,21]          marginal score for left fingertip
- logits_right: [B,21]          marginal score for right fingertip
- S_lr:         [B,21,21]       (left=i, right=j) pairwise compatibility score
- pred_pair:    [B,2]           (i_left*, j_right*) obtained by maximizing joint score
- img_emb:      [B,D]           global image embedding
- node_emb:     [B,21,D]        node embeddings

Keypoint normalization process:
1. Move wrist (joint 0) to origin
2. Fit plane using wrist (0) and finger bases (5,9,13,17), compute palm normal
3. Rotate normal to positive x-axis (palm facing positive x)
4. Global scale normalization (mean distance of all points to origin is 1)

Model architecture:
- DINOv2Backbone: Use pre-trained DINOv2 to extract image features (preserve spatial dimensions)
- HandNodeEncoder: Use graph attention network to model finger bone connection structure + Cross-Attention to fuse images
- PairDecoder: Predict marginal scores for left/right and pairwise compatibility
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


# ------------------------------
# Hand skeleton graph structure definition
# ------------------------------
# Connection relationships of 21 keypoints (bone edges)
# 0: Wrist
# 1-4: Thumb (CMC, MCP, IP, TIP)
# 5-8: Index (MCP, PIP, DIP, TIP)
# 9-12: Middle (MCP, PIP, DIP, TIP)
# 13-16: Ring (MCP, PIP, DIP, TIP)
# 17-20: Pinky (MCP, PIP, DIP, TIP)

HAND_EDGES = [
    # thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # middle finger
    (0, 9), (9, 10), (10, 11), (11, 12),
    # ring finger
    (0, 13), (13, 14), (14, 15), (15, 16),
    # pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm transverse connections (optional, enhance palm structure)
    (5, 9), (9, 13), (13, 17),
]


def build_hand_adjacency_matrix(num_joints: int = 21, edges: list = HAND_EDGES, 
                                  self_loop: bool = True) -> torch.Tensor:
    """Build adjacency matrix for hand skeleton"""
    adj = torch.zeros(num_joints, num_joints)
    for i, j in edges:
        adj[i, j] = 1.0
        adj[j, i] = 1.0  # undirected graph
    if self_loop:
        adj = adj + torch.eye(num_joints)
    # normalization (symmetric normalization)
    deg = adj.sum(dim=1, keepdim=True).clamp(min=1.0)
    adj = adj / deg.sqrt() / deg.sqrt().T
    return adj


# ------------------------------
# DINOv2 Visual Backbone
# ------------------------------
class DINOv2Backbone(nn.Module):
    """
    Use pre-trained DINOv2 to extract image features
    Output: feature map preserving spatial dimensions [B, H', W', D] and global features [B, D]
    """
    def __init__(self, model_name: str = "dinov2_vits14", out_dim: int = 256, 
                 freeze_backbone: bool = True):
        super().__init__()
        self.freeze_backbone = freeze_backbone
        
        # load DINOv2 model
        self.dino = torch.hub.load('facebookresearch/dinov2', model_name)
        print(f"Loaded DINOv2 model: {model_name}")
        
        # DINOv2 feature dimensions
        if 'vits' in model_name:
            dino_dim = 384
        elif 'vitb' in model_name:
            dino_dim = 768
        elif 'vitl' in model_name:
            dino_dim = 1024
        elif 'vitg' in model_name:
            dino_dim = 1536
        else:
            dino_dim = 384  # default
        
        self.dino_dim = dino_dim
        self.out_dim = out_dim
        
        # projection layers
        self.proj = nn.Linear(dino_dim, out_dim)
        self.proj_spatial = nn.Conv2d(dino_dim, out_dim, 1)
        
        # freeze DINOv2 parameters
        if freeze_backbone:
            for param in self.dino.parameters():
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, 3, H, W] input image
        Returns:
            feat_map: [B, H', W', D] spatial feature map
            feat_global: [B, D] global features
        """
        B, C, H, W = x.shape
        
        # DINOv2 expects 224x224 or sizes divisible by 14
        if H != 224 or W != 224:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        with torch.set_grad_enabled(not self.freeze_backbone):
            features = self.dino.forward_features(x)
            
            if isinstance(features, dict):
                patch_tokens = features['x_norm_patchtokens']  # [B, N, D]
                cls_token = features['x_norm_clstoken']  # [B, D]
            else:
                # old version API
                patch_tokens = features[:, 1:, :]  # remove CLS token
                cls_token = features[:, 0, :]
        
        # reshape to spatial dimensions (patch_size=14, 224/14=16)
        h = w = 224 // 14  # = 16
        feat_map = patch_tokens.view(B, h, w, self.dino_dim)  # [B, 16, 16, D]
        
        # projection
        feat_map_proj = self.proj_spatial(feat_map.permute(0, 3, 1, 2))  # [B, out_dim, 16, 16]
        feat_map_proj = feat_map_proj.permute(0, 2, 3, 1)  # [B, 16, 16, out_dim]
        
        feat_global = self.proj(cls_token)  # [B, out_dim]
        
        return feat_map_proj, feat_global


# ------------------------------
# Graph Attention Layer
# ------------------------------
class GraphAttentionLayer(nn.Module):
    """Single-head graph attention layer"""
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Linear(2 * out_dim, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.out_dim = out_dim
    
    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: [B, N, in_dim] node features
            adj: [N, N] adjacency matrix
        Returns:
            h_out: [B, N, out_dim]
        """
        B, N, _ = h.shape
        
        # linear transformation
        Wh = self.W(h)  # [B, N, out_dim]
        
        # compute attention coefficients
        # concatenate features of each node pair
        Wh_repeat_i = Wh.unsqueeze(2).expand(B, N, N, self.out_dim)  # [B, N, N, D]
        Wh_repeat_j = Wh.unsqueeze(1).expand(B, N, N, self.out_dim)  # [B, N, N, D]
        concat = torch.cat([Wh_repeat_i, Wh_repeat_j], dim=-1)  # [B, N, N, 2D]
        
        e = self.leaky_relu(self.a(concat).squeeze(-1))  # [B, N, N]
        
        # compute attention only at adjacent positions
        adj = adj.to(h.device)
        mask = (adj == 0)
        e = e.masked_fill(mask.unsqueeze(0), float('-inf'))
        
        alpha = F.softmax(e, dim=-1)  # [B, N, N]
        alpha = self.dropout(alpha)
        
        # weighted aggregation
        h_out = torch.bmm(alpha, Wh)  # [B, N, out_dim]
        
        return h_out


class MultiHeadGraphAttention(nn.Module):
    """Multi-head graph attention"""
    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert out_dim % num_heads == 0
        self.head_dim = out_dim // num_heads
        self.num_heads = num_heads
        
        self.heads = nn.ModuleList([
            GraphAttentionLayer(in_dim, self.head_dim, dropout) 
            for _ in range(num_heads)
        ])
        self.proj = nn.Linear(out_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # multi-head attention
        head_outputs = [head(h, adj) for head in self.heads]
        h_multi = torch.cat(head_outputs, dim=-1)  # [B, N, out_dim]
        
        # projection + residual
        h_out = self.proj(h_multi)
        h_out = self.dropout(h_out)
        
        # if dimensions match, add residual
        if h.shape[-1] == h_out.shape[-1]:
            h_out = h_out + h
        return self.norm(h_out)


# ------------------------------
# Cross-Attention Module
# ------------------------------
class CrossAttention(nn.Module):
    """
    Cross-Attention: Query from joint features, Key/Value from image feature map
    """
    def __init__(self, d_model: int = 256, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
    
    def forward(self, query: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: [B, N_q, D] joint features (N_q=21)
            kv: [B, N_kv, D] image features (N_kv=H'*W')
        Returns:
            out: [B, N_q, D]
        """
        B, N_q, D = query.shape
        _, N_kv, _ = kv.shape
        
        # projection
        Q = self.q_proj(query).view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(kv).view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(kv).view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        scale = self.head_dim ** -0.5
        attn = torch.matmul(Q, K.transpose(-2, -1)) * scale  # [B, H, N_q, N_kv]
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, V)  # [B, H, N_q, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, N_q, D)
        out = self.out_proj(out)
        
        # residual + LayerNorm
        query = query + self.dropout(out)
        query = self.norm1(query)
        
        # FFN
        query = query + self.ffn(query)
        query = self.norm2(query)
        
        return query


# ------------------------------
# node features encoding + Graph Attention + Cross-Attention
# ------------------------------
class HandNodeEncoder(nn.Module):
    """
    Node encoder:
    1. MLP encode node features
    2. Graph Attention models bone structure
    3. Cross-Attention fuses image features
    4. Transformer self-attention
    """
    def __init__(self, in_dim: int = 26, hidden: int = 256, out_dim: int = 256,
                 num_gat_layers: int = 2, num_cross_attn_layers: int = 2,
                 num_transformer_layers: int = 2, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        # 1. node features MLP
        self.node_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )
        
        # 2. Graph attention layers (model bone connections)
        self.gat_layers = nn.ModuleList([
            MultiHeadGraphAttention(out_dim, out_dim, num_heads=4, dropout=dropout)
            for _ in range(num_gat_layers)
        ])
        
        # 3. Cross-Attention layers (fuse image features)
        self.cross_attn_layers = nn.ModuleList([
            CrossAttention(d_model=out_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_cross_attn_layers)
        ])
        
        # 4. Transformer self-attention layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=out_dim, nhead=num_heads, dim_feedforward=out_dim * 2,
            batch_first=True, dropout=dropout, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        
        # pre-compute adjacency matrix (register as buffer)
        adj = build_hand_adjacency_matrix(21, HAND_EDGES, self_loop=True)
        self.register_buffer('adj', adj)
    
    def forward(self, node_feats: torch.Tensor, img_feat_map: torch.Tensor,
                img_feat_global: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            node_feats: [B, 21, in_dim] node features
            img_feat_map: [B, H', W', D] spatial image features
            img_feat_global: [B, D] global image features (optional)
        Returns:
            H: [B, 21, out_dim]
        """
        B = node_feats.shape[0]
        
        # 1. MLP encoding
        H = self.node_mlp(node_feats)  # [B, 21, D]
        
        # 2. Graph attention (model bone structure)
        for gat in self.gat_layers:
            H = gat(H, self.adj)  # [B, 21, D]
        
        # 3. Cross-Attention (fuse image features)
        # flatten image features into sequence
        H_img, W_img, D = img_feat_map.shape[1], img_feat_map.shape[2], img_feat_map.shape[3]
        img_seq = img_feat_map.view(B, H_img * W_img, D)  # [B, H'*W', D]
        
        for cross_attn in self.cross_attn_layers:
            H = cross_attn(H, img_seq)  # [B, 21, D]
        
        # 4. Transformer self-attention
        H = self.transformer(H)  # [B, 21, D]
        
        return H


# ------------------------------
# Pair Decoder: left/right two queries + pairwise compatibility
# ------------------------------
class PairDecoder(nn.Module):
    """
    Pair Decoder
    
    Function:
    - Predict hand keypoints corresponding to gripper left fingertip (left) and right fingertip (right)
    - No longer predict base point
    
    Output:
    - logits_left:  [B, 21]      marginal score for each keypoint as left fingertip
    - logits_right: [B, 21]      marginal score for each keypoint as right fingertip  
    - S_lr:         [B, 21, 21]  pairwise compatibility matrix S_lr[i,j] represents compatibility of (left=i, right=j)
    - pred_pair:    [B, 2]       final predicted (left_idx, right_idx)
    
    Prediction method:
    - Joint score = logits_left[i] + logits_right[j] + S_lr[i,j]
    - Find the (i, j) pair that maximizes the joint score
    """
    def __init__(self, d_model: int = 256):
        """
        Initialize pair decoder
        
        Args:
            d_model: model hidden dimension, consistent with node embeddings dimension
        """
        super().__init__()
        
        # ====== Two semantic query vectors ======
        # q_left: used to compute marginal score for each keypoint as "left fingertip"
        # q_right: used to compute marginal score for each keypoint as "right fingertip"
        self.q_left = nn.Parameter(torch.randn(d_model))
        self.q_right = nn.Parameter(torch.randn(d_model))
        
        # initialize query vectors
        for q in [self.q_left, self.q_right]:
            nn.init.normal_(q, mean=0.0, std=0.02)

        # ====== Pairwise compatibility bilinear matrix ======
        # W_lr: used to compute left-right pairwise compatibility
        # S_lr[i,j] = H[i] @ W_lr @ H[j].T represents compatibility degree of keypoint i as left, j as right
        self.W_lr = nn.Parameter(torch.empty(d_model, d_model))
        nn.init.xavier_uniform_(self.W_lr)

    def forward(self, H: torch.Tensor):
        """
        forward propagation
        
        Args:
            H: [B, 21, D] node embeddings, output from HandNodeEncoder
            
        Returns:
            logits_left:  [B, 21]     left fingertip marginal score
            logits_right: [B, 21]     right fingertip marginal score
            S_lr:         [B, 21, 21] pairwise compatibility matrix
            pred_pair:    [B, 2]      predicted (left_idx, right_idx)
        """
        B, N, D = H.shape  # B=batch, N=21 keypoints, D=hidden dimension

        # ====== Step 1: Compute marginal scores ======
        # dot product of each keypoint with query vector to get score as left/right
        logits_left  = torch.einsum('bnd,d->bn', H, self.q_left)   # [B, N]
        logits_right = torch.einsum('bnd,d->bn', H, self.q_right)  # [B, N]

        # ====== Step 2: Compute pairwise compatibility ======
        # S_lr[b,i,j] = H[b,i,:] @ W_lr @ H[b,j,:].T
        # represents compatibility degree in sample b of keypoint i as left, j as right
        S_lr = torch.einsum('bnd,de,bme->bnm', H, self.W_lr, H)  # [B, N, N]

        # ====== Step 3: Joint scoring to find optimal pairing ======
        # comb[i,j] = logits_left[i] + logits_right[j] + S_lr[i,j]
        # allow i=j (same keypoint as both left and right fingertips)
        comb = (
            logits_left[:, :, None] +     # [B, N, 1]
            logits_right[:, None, :] +    # [B, 1, N]
            S_lr                          # [B, N, N]
        )  # [B, N, N]

        # flatten and find max index
        comb_flat = comb.view(B, -1)             # [B, N^2]
        idx = torch.argmax(comb_flat, dim=1)     # [B]
        
        # recover from flattened index (i_left, j_right)
        i_left = idx // N      # [B]
        j_right = idx % N      # [B]
        pred_pair = torch.stack([i_left, j_right], dim=1)  # [B, 2]

        return logits_left, logits_right, S_lr, pred_pair


# ------------------------------
# Top-level Model
# ------------------------------
class Hand2GripperModel(nn.Module):
    """
    Hand2Gripper main model (Left/Right Only)
    
    Function:
    - Map hand 3D keypoints to gripper left and right fingertips
    - Use DINOv2 pre-trained backbone to extract image features
    - Use graph attention network to model finger bone structure
    - Use Cross-Attention to fuse image features
    
    Architecture:
    1. DINOv2Backbone: extract image features [B,H',W',D] and global features [B,D]
    2. HandNodeEncoder: encode joint features, fuse images, output node embeddings [B,21,D]
    3. PairDecoder: predict left/right keypoint indices
    
    Input:
    - img_crop:     [B, 3, S, S]   cropped hand image
    - keypoints_3d: [B, 21, 3]     3D keypoint coordinates
    - contact:      [B, 21]        contact probability (retained but not used)
    - is_right:     [B]            right hand flag
    
    Output:
    - logits_left:  [B, 21]        left fingertip marginal score
    - logits_right: [B, 21]        right fingertip marginal score
    - S_lr:         [B, 21, 21]    pairwise compatibility matrix
    - pred_pair:    [B, 2]         predicted (left_idx, right_idx)
    - img_emb:      [B, D]         global image embedding
    - node_emb:     [B, 21, D]     node embeddings
    """
    def __init__(self, d_model: int = 256, img_size: int = 256,
                 backbone: str = "dinov2_vits14", freeze_backbone: bool = True):
        """
        initialize model
        
        Args:
            d_model: model hidden dimension
            img_size: cropped image size
            backbone: DINOv2 backbone model name ("dinov2_vits14", "dinov2_vitb14", etc.)
            freeze_backbone: whether to freeze DINOv2 parameters
        """
        super().__init__()
        self.img_size = img_size
        self.crop_scale = 1.2  # bbox expansion ratio
        
        # ====== Visual Backbone ======
        # use pre-trained DINOv2 to extract image features
        self.backbone = DINOv2Backbone(
            model_name=backbone, out_dim=d_model, freeze_backbone=freeze_backbone
        )
        
        # ====== Node Encoder ======
        # fuse 3D keypoint features and image features
        self.encoder = HandNodeEncoder(
            in_dim=26,                    # 3(xyz) + 1(contact) + 21(onehot) + 1(is_right)
            hidden=d_model, 
            out_dim=d_model,
            num_gat_layers=2,             # number of graph attention layers
            num_cross_attn_layers=2,      # number of Cross-Attention layers  
            num_transformer_layers=2,     # number of Transformer self-attention layers
            num_heads=8, 
            dropout=0.1
        )
        
        # ====== Pair Decoder ======
        # predict left/right keypoints
        self.decoder = PairDecoder(d_model=d_model)
    
    def _load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            self.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            self.load_state_dict(checkpoint, strict=False)

    @staticmethod
    def _normalize_keypoints_xyz(kp3d: torch.Tensor) -> torch.Tensor:
        """
        Hand pose normalization:
        1. Move wrist (0) to origin
        2. Fit plane using palm root (0) and four finger bases (5,9,13,17), compute normal vector
        3. Rotate normal to positive x-axis (palm facing positive x)
        4. Global scale normalization
        """
        kp = kp3d.clone()
        B = kp.shape[0]
        device = kp.device
        dtype = kp.dtype

        # Step 1: Center palm root
        wrist = kp[:, 0:1, :]  # [B,1,3]
        kp = kp - wrist

        # Step 2: Extract 5 keypoints to fit plane
        palm_indices = [0, 5, 9, 13, 17]  # palm root + four finger MCP joints
        palm_pts = kp[:, palm_indices, :]  # [B,5,3]

        # compute center and de-center
        palm_center = palm_pts.mean(dim=1, keepdim=True)  # [B,1,3]
        palm_centered = palm_pts - palm_center  # [B,5,3]

        # SVD find eigenvector corresponding to smallest eigenvalue as normal vector
        U, S, Vh = torch.linalg.svd(palm_centered, full_matrices=False)  # Vh: [B,3,3]
        normal = Vh[:, 2, :]  # right singular vector corresponding to smallest singular value, i.e., normal vector [B,3]

        # Step 3: Ensure normal vector points towards palm
        finger_dir = kp[:, 9, :] - kp[:, 0, :]  # [B,3]
        finger_dir = finger_dir / (finger_dir.norm(dim=1, keepdim=True) + 1e-8)

        up_ref = kp[:, 10, :] - kp[:, 9, :]  # [B,3]
        up_ref = up_ref / (up_ref.norm(dim=1, keepdim=True) + 1e-8)

        expected_normal = torch.cross(finger_dir, up_ref, dim=1)  # [B,3]
        dot = (normal * expected_normal).sum(dim=1, keepdim=True)  # [B,1]
        normal = normal * torch.sign(dot + 1e-8)

        # Step 4: Build rotation matrix
        target = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
        target = target.unsqueeze(0).expand(B, 3)

        normal = normal / (normal.norm(dim=1, keepdim=True) + 1e-8)

        axis = torch.cross(normal, target, dim=1)
        axis_norm = axis.norm(dim=1, keepdim=True) + 1e-8
        axis = axis / axis_norm

        cos_theta = (normal * target).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
        sin_theta = axis_norm.squeeze(-1).unsqueeze(-1)

        is_parallel = (axis_norm.squeeze(-1) < 1e-6)
        is_same_dir = (cos_theta.squeeze(-1) > 0)

        K = torch.zeros(B, 3, 3, device=device, dtype=dtype)
        K[:, 0, 1] = -axis[:, 2]
        K[:, 0, 2] = axis[:, 1]
        K[:, 1, 0] = axis[:, 2]
        K[:, 1, 2] = -axis[:, 0]
        K[:, 2, 0] = -axis[:, 1]
        K[:, 2, 1] = axis[:, 0]

        I = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(B, 3, 3)
        R = I + sin_theta.unsqueeze(-1) * K + (1 - cos_theta).unsqueeze(-1) * (K @ K)

        for b in range(B):
            if is_parallel[b]:
                if is_same_dir[b]:
                    R[b] = torch.eye(3, device=device, dtype=dtype)
                else:
                    R[b] = torch.diag(torch.tensor([-1.0, 1.0, -1.0], device=device, dtype=dtype))

        kp = torch.bmm(kp, R.transpose(1, 2))

        dist = torch.norm(kp, dim=-1)
        scale = dist.mean(dim=1, keepdim=True).clamp(min=1e-6)
        kp = kp / scale.unsqueeze(-1)

        return kp

    @staticmethod
    def _build_node_features(kp_xyz_norm: torch.Tensor, contact: torch.Tensor, 
                              is_right: torch.Tensor) -> torch.Tensor:
        """Concatenate [xyz | contact | onehot | is_right] -> [B,21,26]"""
        B, N, _ = kp_xyz_norm.shape
        onehot = torch.eye(N, device=kp_xyz_norm.device).unsqueeze(0).repeat(B, 1, 1)
        contact_f = contact.unsqueeze(-1)
        isr = is_right.view(B, 1, 1).repeat(1, N, 1).float()
        feats = torch.cat([kp_xyz_norm, contact_f, onehot, isr], dim=-1)
        return feats

    @staticmethod
    def _expand_bbox(bbox: torch.Tensor, H: int, W: int, scale: float = 1.2) -> torch.Tensor:
        x1, y1, x2, y2 = bbox.unbind(dim=1)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = (x2 - x1).clamp(min=1.0)
        h = (y2 - y1).clamp(min=1.0)
        w2 = w * scale / 2.0
        h2 = h * scale / 2.0
        nx1 = (cx - w2).clamp(min=0.0, max=W - 1.0)
        ny1 = (cy - h2).clamp(min=0.0, max=H - 1.0)
        nx2 = (cx + w2).clamp(min=0.0, max=W - 1.0)
        ny2 = (cy + h2).clamp(min=0.0, max=H - 1.0)
        return torch.stack([nx1, ny1, nx2, ny2], dim=1)
    
    def _read_color(self, color: np.ndarray) -> torch.Tensor:
        """
        Input: color can be [H,W,3] (HWC) or [3,H,W] (CHW), uint8 or float32/64
        Output: [1,3,H,W], float32, range [0,1]
        """
        if color.ndim != 3:
            raise ValueError(f"color ndim={color.ndim}, needs 3 dimensions")
        # convert to float32 and normalize
        if color.dtype == np.uint8:
            color = color.astype(np.float32) / 255.0
        else:
            color = color.astype(np.float32, copy=False)

        if color.shape[0] == 3 and color.ndim == 3:       # CHW
            chw = color
        elif color.shape[-1] == 3:                        # HWC -> CHW
            chw = np.transpose(color, (2, 0, 1))
        else:
            raise ValueError(f"color shape invalid: {color.shape}, expect CHW or HWC with 3 channels")

        chw = np.ascontiguousarray(chw)
        t = torch.from_numpy(chw).unsqueeze(0)            # [1,3,H,W]
        return t.float()


    def _read_bbox(self, bbox: np.ndarray) -> torch.Tensor:
        """
        Input: bbox [4] (x1,y1,x2,y2), any numeric type
        Output: [1,4] float32
        """
        bbox = np.asarray(bbox).astype(np.float32, copy=False)
        if bbox.shape != (4,):
            raise ValueError(f"bbox shape should be (4,), actual {bbox.shape}")
        return torch.from_numpy(bbox).unsqueeze(0).float()  # [1,4]


    def _read_keypoints_3d(self, keypoints_3d: np.ndarray) -> torch.Tensor:
        """
        Input: [21,3] or [1,21,3]
        Output: [1,21,3] float32
        """
        kp = np.asarray(keypoints_3d)
        if kp.ndim == 2:
            if kp.shape != (21, 3):
                raise ValueError(f"keypoints_3d shape should be (21,3), actual {kp.shape}")
            kp = kp[None, ...]                             # -> [1,21,3]
        elif kp.ndim == 3:
            if kp.shape[1:] != (21, 3) and not (kp.shape[0:3] == (1, 21, 3)):
                raise ValueError(f"unsupported keypoints_3d shape: {kp.shape}")
        else:
            raise ValueError(f"keypoints_3d ndim should be 2 or 3, actual {kp.ndim}")

        kp = kp.astype(np.float32, copy=False)
        return torch.from_numpy(kp).float()               # [1,21,3]


    def _read_contact(self, contact: np.ndarray) -> torch.Tensor:
        """
        Input: [21] or [1,21]
        Output: [1,21] float32
        """
        c = np.asarray(contact)
        if c.ndim == 1:
            if c.shape != (21,):
                raise ValueError(f"contact shape should be (21,), actual {c.shape}")
            c = c[None, ...]                               # -> [1,21]
        elif c.ndim == 2 and c.shape[0] == 1 and c.shape[1] == 21:
            pass
        else:
            raise ValueError(f"unsupported contact shape: {c.shape}")
        c = c.astype(np.float32, copy=False)
        return torch.from_numpy(c).float()                # [1,21]


    def _read_is_right(self, is_right: np.ndarray) -> torch.Tensor:
        """
        Input: scalar, [1], or [B] of 0/1
        Output: [1] long (internally converted to float and concatenated to features)
        """
        ir = np.asarray(is_right)
        if ir.ndim == 0:
            ir = ir[None]                                  # -> [1]
        elif ir.ndim == 1 and ir.shape[0] == 1:
            pass
        else:
            raise ValueError(f"is_right shape should be scalar or (1,), actual {ir.shape}")
        ir = ir.astype(np.int64, copy=False)
        return torch.from_numpy(ir)                        # [1], long


    def _crop_and_resize(self, color: torch.Tensor, bbox: torch.Tensor) -> torch.Tensor:
        B, C, H, W = color.shape
        bbox = self._expand_bbox(bbox, H, W, self.crop_scale)
        crops = []
        for b in range(B):
            x1, y1, x2, y2 = bbox[b]
            x1i = int(torch.floor(x1).item())
            y1i = int(torch.floor(y1).item())
            x2i = int(torch.ceil(x2).item())
            y2i = int(torch.ceil(y2).item())
            x2i = max(x2i, x1i + 1)
            y2i = max(y2i, y1i + 1)
            patch = color[b:b+1, :, y1i:y2i, x1i:x2i]
            patch = F.interpolate(patch, size=(self.img_size, self.img_size), 
                                   mode="bilinear", align_corners=False)
            crops.append(patch)
        crop_img = torch.cat(crops, dim=0)
        return crop_img  # [B,3,S,S]

    def forward(self, img_crop: torch.Tensor, keypoints_3d: torch.Tensor,
                contact: torch.Tensor, is_right: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        model forward propagation
        
        Args:
            img_crop:     [B, 3, S, S]   cropped and resized hand image
            keypoints_3d: [B, 21, 3]     hand 3D keypoint coordinates
            contact:      [B, 21]        contact probability (interface retained but not used)
            is_right:     [B]            whether right hand (1=right, 0=left)
            
        Returns:
            dict: containing the following keys:
                - logits_left:  [B, 21]      left fingertip marginal score
                - logits_right: [B, 21]      right fingertip marginal score
                - S_lr:         [B, 21, 21]  pairwise compatibility matrix
                - pred_pair:    [B, 2]       predicted (left_idx, right_idx)
                - img_emb:      [B, D]       global image embedding
                - node_emb:     [B, 21, D]   node embeddings
        """
        # ====== Step 1: Extract Image Features ======
        # img_feat_map: [B, H', W', D] spatial feature map, used for Cross-Attention
        # img_emb: [B, D] global features, used for output
        img_feat_map, img_emb = self.backbone(img_crop)

        # ====== Step 2: Keypoint Normalization and Node Features Construction ======
        # normalize 3D keypoints to unified coordinate system
        kp_xyz_norm = self._normalize_keypoints_xyz(keypoints_3d)
        # concatenate node features: [xyz | contact | onehot | is_right]
        node_feats = self._build_node_features(kp_xyz_norm, contact, is_right)

        # ====== Step 3: Encoding ======
        # get node embeddings through graph attention + Cross-Attention + Transformer
        H = self.encoder(node_feats, img_feat_map, img_emb)  # [B, 21, D]

        # ====== Step 4: Decoding ======
        # predict left/right keypoint indices
        logits_left, logits_right, S_lr, pred_pair = self.decoder(H)

        return {
            "logits_left": logits_left,     # [B, 21] left fingertip marginal score
            "logits_right": logits_right,   # [B, 21] right fingertip marginal score
            "S_lr": S_lr,                   # [B, 21, 21] pairwise compatibility
            "pred_pair": pred_pair,         # [B, 2] predicted (left, right)
            "img_emb": img_emb,             # [B, D] image embedding
            "node_emb": H,                  # [B, 21, D] node embeddings
        }


# ------------------------------
# Visualization Tools
# ------------------------------
def visualize_hand_keypoints(kp_before: np.ndarray, kp_after: np.ndarray, 
                              title: str = "Hand Keypoints Normalization",
                              save_path: str = None):
    """
    Visualize comparison of hand keypoints before and after normalization
    
    Args:
        kp_before: [21, 3] keypoints before normalization
        kp_after:  [21, 3] keypoints after normalization
        title: chart title
        save_path: save path, None to display
    """
    import matplotlib.pyplot as plt
    
    finger_links = [
        [0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12],
        [0, 13, 14, 15, 16], [0, 17, 18, 19, 20],
    ]
    finger_colors = ['red', 'orange', 'green', 'blue', 'purple']
    
    fig = plt.figure(figsize=(14, 6))
    
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title('Before Normalization')
    for finger_idx, links in enumerate(finger_links):
        pts = kp_before[links]
        ax1.plot(pts[:, 0], pts[:, 1], pts[:, 2], 'o-', 
                 color=finger_colors[finger_idx], linewidth=2, markersize=4)
    ax1.scatter(*kp_before[0], color='black', s=100, marker='*', label='Wrist')
    palm_idx = [0, 5, 9, 13, 17]
    palm_pts = kp_before[palm_idx]
    ax1.scatter(palm_pts[:, 0], palm_pts[:, 1], palm_pts[:, 2], 
                color='cyan', s=60, marker='s', label='Palm plane')
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    ax1.legend()
    _set_axes_equal(ax1)
    
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title('After Normalization (Palm normal → +X)')
    for finger_idx, links in enumerate(finger_links):
        pts = kp_after[links]
        ax2.plot(pts[:, 0], pts[:, 1], pts[:, 2], 'o-', 
                 color=finger_colors[finger_idx], linewidth=2, markersize=4)
    ax2.scatter(*kp_after[0], color='black', s=100, marker='*', label='Wrist (origin)')
    palm_pts = kp_after[palm_idx]
    ax2.scatter(palm_pts[:, 0], palm_pts[:, 1], palm_pts[:, 2], 
                color='cyan', s=60, marker='s', label='Palm plane')
    ax2.quiver(0, 0, 0, 0.5, 0, 0, color='r', arrow_length_ratio=0.1, linewidth=2)
    ax2.quiver(0, 0, 0, 0, 0.5, 0, color='g', arrow_length_ratio=0.1, linewidth=2)
    ax2.quiver(0, 0, 0, 0, 0, 0.5, color='b', arrow_length_ratio=0.1, linewidth=2)
    ax2.text(0.55, 0, 0, '+X (palm normal)', fontsize=8)
    ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
    ax2.legend()
    _set_axes_equal(ax2)
    
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    plt.close()


def _set_axes_equal(ax):
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])


# ------------------------------
# Demo
# ------------------------------
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hand2Gripper Model Demo")
    parser.add_argument("--npz", type=str, default="", help="Path to .npz sample file")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to model checkpoint")
    parser.add_argument("--save_vis", type=str, default=None, help="Path to save visualization")
    parser.add_argument("--use_random", action="store_true", help="Use random data instead of npz file")
    parser.add_argument("--no_freeze_backbone", action="store_true", help="do not freeze DINOv2 parameters")
    args = parser.parse_args()
    
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # initialize model (default uses DINOv2 and freezes parameters)
    model = Hand2GripperModel(
        d_model=256, img_size=256, 
        freeze_backbone=not args.no_freeze_backbone
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # load checkpoint
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}")
        model._load_checkpoint(args.checkpoint)
        print("Checkpoint loaded successfully")
    
    model.eval()

    if args.npz and os.path.exists(args.npz) and not args.use_random:
        print(f"Loading data from {args.npz}")
        data = np.load(args.npz, allow_pickle=True)
        
        img_rgb = data["img_rgb"]
        bbox = data["bbox"]
        kpts_3d = data["kpts_3d"]
        contact_logits = data["contact_logits"]
        is_right = data["is_right"]
        
        has_gt = "selected_gripper_blr_ids" in data
        if has_gt:
            gt_blr = data["selected_gripper_blr_ids"]
            print(f"Ground truth (base, left, right): {gt_blr}")
        
        img_rgb_t = model._read_color(img_rgb).to(device)
        bbox_t = model._read_bbox(bbox).to(device)
        kpts_3d_t = model._read_keypoints_3d(kpts_3d).to(device)
        contact_t = model._read_contact(contact_logits).to(device)
        is_right_t = model._read_is_right(is_right).to(device)
        
    else:
        print("Using random data for demo...")
        H, W = 480, 640
        
        img_rgb = np.random.rand(H, W, 3).astype(np.float32)
        bbox = np.array([120, 80, 320, 360], dtype=np.int32)
        kpts_3d = np.random.randn(21, 3).astype(np.float32) * 0.05
        contact_logits = np.random.rand(21).astype(np.float32)
        is_right = np.array([1], dtype=np.int64)
        
        has_gt = False
        
        img_rgb_t = model._read_color(img_rgb).to(device)
        bbox_t = model._read_bbox(bbox).to(device)
        kpts_3d_t = model._read_keypoints_3d(kpts_3d).to(device)
        contact_t = model._read_contact(contact_logits).to(device)
        is_right_t = model._read_is_right(is_right).to(device)

    # ===== Visualize Normalization Effect =====
    print("="*80)
    print("Visualizing keypoints normalization...")
    kp_before = kpts_3d if kpts_3d.ndim == 2 else kpts_3d[0]  # [21,3]
    kp_after = model._normalize_keypoints_xyz(kpts_3d_t).cpu().numpy()[0]  # [21,3]
    visualize_hand_keypoints(
        kp_before, kp_after,
        title="Hand Keypoints Normalization",
        save_path=args.save_vis
    )

    # ===== Model Inference =====
    print("="*80)
    print("Running model inference...")
    with torch.no_grad():
        crop_t = model._crop_and_resize(img_rgb_t, bbox_t)  # [1,3,256,256]
        out = model(crop_t, kpts_3d_t, contact_t, is_right_t)

    pred_pair = out["pred_pair"].cpu().numpy()[0]  # [2]
    print(f"Predicted (left, right): {pred_pair}")
    
    if has_gt:
        gt_lr = gt_blr[1:]  # take left, right part
        print(f"Ground truth (left, right): {gt_lr}")
        match = np.array_equal(pred_pair, gt_lr)
        print(f"Match: {match}")

    # print top-3 for each logit
    print("-"*80)
    print("Top-3 predictions for each role:")
    for role, key in [("Left", "logits_left"), ("Right", "logits_right")]:
        logits = out[key].cpu().numpy()[0]  # [21]
        top3_idx = np.argsort(logits)[::-1][:3]
        top3_scores = logits[top3_idx]
        print(f"  {role}: {list(zip(top3_idx.tolist(), top3_scores.tolist()))}")

    print("-"*80)
    print(f"img_emb shape: {out['img_emb'].shape}")
    print(f"node_emb shape: {out['node_emb'].shape}")
    print("Done.")