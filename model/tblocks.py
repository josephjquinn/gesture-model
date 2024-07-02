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
