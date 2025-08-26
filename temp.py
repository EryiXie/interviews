import math
import torch
import torch.nn as nn

class MHSA(nn.Module):
    def __init__(self, dim, num_heads=8, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=True)  # W_o
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x)                       # [B, N, 3D]
        q, k, v = qkv.chunk(3, dim=-1)         # each [B, N, D]
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B,H,N,d]
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)      # [B,H,N,N]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ v                                                   # [B,H,N,d]
        out = out.transpose(1, 2).reshape(B, N, D)                       # [B,N,D]
        out = self.proj(out)                                             # W_o
        out = self.proj_drop(out)
        return out

# ---- From your snippet ----
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_ch=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_ch, embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size)

    def forward(self, x):  # x: [B,3,H,W]
        x = self.proj(x)                                # [B, D, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)                # [B, N, D], N=(H/P)*(W/P)
        return x

class ViTBlock(nn.Module):
    def __init__(self, dim, heads=8, mlp_ratio=4.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, int(dim*mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim*mlp_ratio), dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

# ---- ViT Encoder ----
class ViTEncoder(nn.Module):
    """
    Multi-block ViT encoder with:
      - Patch embedding
      - Learnable positional embeddings (interpolated if input size changes)
      - Optional [CLS] token
      - Returns final tokens and selected hidden states for decoders (e.g., DPT)

    Args:
        img_size: default size used to init pos_embed (can still run with other sizes)
        patch_size: patch size (kernel/stride for Conv2d)
        in_ch: input channels
        embed_dim: token dimension
        depth: number of transformer blocks
        heads: attention heads per block
        mlp_ratio: FFN expansion ratio
        drop_rate: dropout after adding pos embeddings
        use_cls_token: prepend a learnable [CLS] token (True = returns cls + patch tokens)
        out_indices: list of block indices (1-based) whose outputs you also return
                     e.g., [3, 6, 9, 12] for a 12-block encoder
    """
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
        self.use_cls_token = use_cls_token
        self.out_indices = set(out_indices)

        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_ch, embed_dim)

        # Positional embeddings (1 x (N + cls) x D) for default img_size
        num_patches = self.patch_embed.num_patches
        pos_len = num_patches + (1 if use_cls_token else 0)
        self.pos_embed = nn.Parameter(torch.zeros(1, pos_len, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Optional [CLS] token
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        else:
            self.cls_token = None

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            ViTBlock(embed_dim, heads=heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

    @torch.no_grad()
    def _interpolate_pos_encoding(self, x, H_p, W_p):
        """
        x: [B, N(+1), D], H_p/W_p: patch grid for the current input
        Interpolate pos_embed if input resolution differs from initialization.
        """
        N = H_p * W_p
        if self.use_cls_token:
            cls_pos = self.pos_embed[:, :1]          # [1,1,D]
            patch_pos = self.pos_embed[:, 1:]        # [1,N0,D]
        else:
            cls_pos = None
            patch_pos = self.pos_embed               # [1,N0,D]

        N0 = patch_pos.shape[1]
        if N0 == N:
            # same spatial size → no interpolation needed
            if self.use_cls_token:
                return torch.cat([cls_pos, patch_pos], dim=1)
            return patch_pos

        # reshape to (1, D, H0, W0)
        D = patch_pos.shape[-1]
        H0 = W0 = int(math.sqrt(N0))
        patch_pos_2d = patch_pos.reshape(1, H0, W0, D).permute(0, 3, 1, 2)
        patch_pos_2d = torch.nn.functional.interpolate(
            patch_pos_2d, size=(H_p, W_p), mode="bicubic", align_corners=False
        )
        patch_pos_new = patch_pos_2d.permute(0, 2, 3, 1).reshape(1, N, D)

        if self.use_cls_token:
            return torch.cat([cls_pos, patch_pos_new], dim=1)
        else:
            return patch_pos_new

    def forward(self, x):
        """
        Returns:
            tokens: [B, N(+1), D] final encoded tokens (LayerNorm'ed)
            hiddens: list of tensors from blocks in out_indices (post-block), each [B, N(+1), D]
                     (useful for multi-scale decoders like DPT)
        """
        B, _, H, W = x.shape
        # Patchify to tokens
        x = self.patch_embed(x)  # [B, N, D] with N=(H/P)*(W/P)
        H_p, W_p = H // self.patch_embed.patch_size, W // self.patch_embed.patch_size

        # Add cls token (optional)
        if self.use_cls_token:
            cls_tok = self.cls_token.expand(B, -1, -1)   # [B,1,D]
            x = torch.cat([cls_tok, x], dim=1)           # [B,1+N,D]

        # Add (interpolated) positional embeddings + dropout
        pos = self._interpolate_pos_encoding(x, H_p, W_p)  # [B, 1+N or N, D]
        x = x + pos
        x = self.pos_drop(x)

        # Pass through blocks, collect chosen hidden states
        hiddens = []
        for i, blk in enumerate(self.blocks, start=1):
            x = blk(x)
            if i in self.out_indices:
                hiddens.append(x)

        # Final norm
        x = self.norm(x)
        return x, hiddens

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    2D sine-cosine positional embeddings.

    Args:
        embed_dim: total embedding dimension (must be even).
        grid_size: int or tuple (H, W) for patch grid.
        cls_token: if True, prepend a [CLS] embedding (zeros).

    Returns:
        pos_embed: [1, N(+1), D]
    """
    if isinstance(grid_size, int):
        H = W = grid_size
    else:
        H, W = grid_size

    # Create grid of positions
    grid_y = torch.arange(H, dtype=torch.float32)
    grid_x = torch.arange(W, dtype=torch.float32)
    grid = torch.meshgrid(grid_y, grid_x, indexing="ij")  # [2, H, W]
    grid = torch.stack(grid, dim=0).reshape(2, -1)        # [2, N]
    # grid[0] = y coords, grid[1] = x coords

    # Encode row and col separately
    assert embed_dim % 2 == 0, "embed_dim must be even"
    half_dim = embed_dim // 2
    pos_y = get_1d_sincos_pos_embed_from_grid(half_dim, grid[0])  # [N, D/2]
    pos_x = get_1d_sincos_pos_embed_from_grid(half_dim, grid[1])  # [N, D/2]
    pos = torch.cat([pos_y, pos_x], dim=1)  # [N, D]

    if cls_token:
        cls = torch.zeros([1, 1, embed_dim], dtype=torch.float32)
        pos = torch.cat([cls.squeeze(0), pos], dim=0)

    return pos.unsqueeze(0)  # [1, N(+1), D]

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    1D sine-cosine positional encoding.
    pos: [N]
    Returns: [N, D]
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega = 1.0 / (10000 ** (2 * omega / embed_dim))  # [D/2]
    out = pos[:, None] * omega[None, :]               # [N, D/2]
    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)
    return torch.cat([emb_sin, emb_cos], dim=1)

# ---------------- Example usage ----------------
if __name__ == "__main__":
    B, C, H, W = 2, 3, 256, 256
    img = torch.randn(B, C, H, W)

    encoder = ViTEncoder(
        img_size=224,          # init size for pos_embed; works with other sizes via interpolation
        patch_size=16,
        in_ch=3,
        embed_dim=384,
        depth=12,
        heads=6,
        mlp_ratio=4.0,
        drop_rate=0.1,
        use_cls_token=True,
        out_indices=(2, 4, 6, 8),
    )

    tokens, hiddens = encoder(img)
    print(tokens.shape)            # [B, N+1, D]
    for k, h in enumerate(hiddens, 1):
        print(f"hidden[{k}]:", h.shape)
    total_params = sum(p.numel() for p in encoder.parameters())

    print(f"Number of parameters: {total_params}")
