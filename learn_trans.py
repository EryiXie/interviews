import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(
        self, 
        img_size=224, 
        patch_size=16,
        in_channels=3,
        embed_dims=384 
    ):
        super().__init__()
        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dims,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.img_size = img_size
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.grid_size = (img_size // patch_size, img_size // patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        x = self.projection(x) # Bx3xHxW -> BxDxhxw
        x = x.flatten(start_dim=2).transpose(1, 2) # BxDxhxw -> BxNxD
        return x


class ViTBlock(nn.Module):
    def __init__(
        self,
        embed_dims=384,
        num_heads=12,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dims)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dims, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dims, int(embed_dims*mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dims*mlp_ratio), embed_dims)
        )

    
    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class ViTEncoder(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_ch=3,
        embed_dim=768,
        depth=12,
        heads=12,
        mlp_ratio=4.0,
        drop_rate=0.0,
        use_cls_token=False,
        out_indices=(3, 6, 9, 12),
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_cls_tocken = use_cls_token
        self.num_patches = self.patch_embedding.num_patches
        self.out_indices = set(out_indices)

        self.patch_embedding = PatchEmbedding(img_size=img_size, patch_size=patch_size, in_channels=in_ch, embed_dims=embed_dim)
        
        # Positional Emdedding
        num_patches = self.num_patches
        pos_len = num_patches if not use_cls_token else num_patches+1
        self.pos_embeddings = nn.Parameter(torch.zeros(1, pos_len, embed_dim))
        nn.init.trunc_normal_(self.pos_embeddings, std=0.02)

        # Optinal CLS
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        else:
            self.cls_token = None
        
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Transformer blocks
        self.blocks = nn.Sequential(
            ViTBlock(embed_dim, heads, mlp_ratio)
            for _ in range(depth)
        )

        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        x = self.patch_embedding(x)
        if self.use_cls_tocken:
            cls_token = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_token, x], dim=1)
        
        pos = self.pos_embeddings.expand(B, -1, -1) #TODO: Not Correct

        x = x + pos
        x = self.pos_drop(x)

        hiddens=[]
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.out_indices:
                hiddens.append(x)
        x = self.norm(x)
        return x