import os
from PIL import Image
import torch
import random
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class gestureDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        label_dirs = sorted(os.listdir(root_dir))

        for label_idx, label_name in enumerate(label_dirs):
            label_dir = os.path.join(root_dir, label_name)
            if not os.path.isdir(label_dir):
                continue

            for img_name in os.listdir(label_dir):
                if img_name.endswith(("png", "jpg", "jpeg", "bmp", "tiff")):
                    img_path = os.path.join(label_dir, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(label_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        return image, label

    def __getshape__(self):
        return (self.__len__(), *self.__getitem__(0)[0].shape)

    def display_random_images(self, n, classes=None, display_shape=False):
        random_samples_idx = random.sample(range(len(self)), k=n)

        grid_size = int(n**0.5)
        if grid_size**2 < n:
            grid_size += 1

        fig, axes = plt.subplots(grid_size, grid_size, figsize=(16, 16))
        axes = axes.flatten()

        for i, targ_sample in enumerate(random_samples_idx):
            targ_image, targ_label = self[targ_sample]

            targ_image_adjust = targ_image.permute(1, 2, 0)

            ax = axes[i]
            ax.imshow(targ_image_adjust)
            ax.axis("off")
            title = (
                f"class: {classes[targ_label]}" if classes else f"class: {targ_label}"
            )
            if display_shape:
                title += f"\nshape: {targ_image_adjust.shape}"
            ax.set_title(title, fontsize=8)

        for j in range(i + 1, grid_size**2):
            axes[j].axis("off")

        plt.tight_layout()
        plt.show()

    def show_one_image_per_directory(dataset, num_dirs=27, rows=3, cols=9):
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        axes = axes.flatten()

        seen_labels = set()
        count = 0
        for idx in range(len(dataset)):
            if count >= num_dirs:
                break
            image, label = dataset[idx]
            if label not in seen_labels:
                seen_labels.add(label)
                image = image.permute(1, 2, 0)
                axes[count].imshow(image)
                axes[count].set_title(f"Label: {label}", fontsize=8)
                axes[count].axis("off")
                count += 1

        for j in range(count, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        plt.show()
