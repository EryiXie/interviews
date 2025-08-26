import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from random import random

from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import Resize, ToTensor
from torchvision.transforms.functional import to_pil_image
from einops.layers.torch import Rearrange

to_tensor = [Resize((256, 256)), ToTensor()]

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, image, target):
        for t in self.transforms:
            image = t(image)
        return image, target
    

def show_images(images, num_samples=40, cols=8):
    """ Plots some samples from the dataset """
    plt.figure(figsize=(15,15))
    idx = int(len(dataset) / num_samples)
    print(images)
    for i, img in enumerate(images):
        if i % idx == 0:
            plt.subplot(int(num_samples/cols) + 1, cols, int(i/idx) + 1)
            plt.imshow(to_pil_image(img[0]))




# 200 images for each pet
dataset = OxfordIIITPet(root=".", download=True, transforms=Compose(to_tensor))
#show_images(dataset)

class PatchEmbedding_withMLP(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, emb_size=128):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_size*patch_size*in_channels, emb_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)
        return x

class PatchEmbedding_withConv(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, embed_dim=128):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size)
    
    def forward(self, x):
        # [B,3,H,W] -> [B,embed_dim,N]
        x = self.proj(x)          # [B,D,H/P,W/P]
        x = x.flatten(2).transpose(1,2)  # [B,N,D]
        return x

# Run a quick test
sample_datapoint = torch.unsqueeze(dataset[0][0], 0)
print("Initial shape: ", sample_datapoint.shape)
embedding = PatchEmbedding_withConv()(sample_datapoint)
print("Patches shape: ", embedding.shape)


class ViTBlock(nn.Module):
    def __init__(self, dim, n_heads=8, dropout=True, mlp_ratio=4.):
        super().__init__()
        self.n_heads = n_heads
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim*mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim*mlp_ratio), dim)
        )


    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

