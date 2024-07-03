import torch.nn as nn
import torch


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16):
        super().__init__()
        self.patch_size = patch_size
        self.patch = nn.Conv2d(
            in_channels=in_channels,
            out_channels=patch_size * patch_size * in_channels,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )

    def forward(self, x):
        self.flatten = nn.Flatten(
            start_dim=2,
            end_dim=3,
        )

        image_resolution = x.shape[-1]
        assert (
            image_resolution % self.patch_size == 0
        ), f"Input image size must be divisble by patch size, image shape: {image_resolution}, patch size: {self.patch_size}"

        x_patched = self.patch(x)
        x_flattened = torch.flatten(x_patched, start_dim=2, end_dim=3)
        return x_flattened.permute(0, 2, 1)


class MultiheadSelfAttentionBlock(nn.Module):
    def __init__(
        self, embedding_dim: int = 768, num_heads: int = 12, attn_dropout: float = 0
    ):
        super().__init__()

        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )

    def forward(self, x):
        x = self.layer_norm(x)
        attn, _ = self.multihead_attn(query=x, key=x, value=x, need_weights=False)
        return attn


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 768,  # Hidden Size D
        mlp_size: int = 3072,  # MLP size
        dropout: float = 0.1,
    ):
        super().__init__()

        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=mlp_size),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(
                in_features=mlp_size,
                out_features=embedding_dim,
            ),
            nn.Dropout(p=dropout),
        )

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x
