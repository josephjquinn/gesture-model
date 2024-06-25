import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from matplotlib import pyplot as plt


class gestureDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images, structured with subfolders.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for label in range(27):
            label_dir = os.path.join(root_dir, str(label))
            if not os.path.isdir(label_dir):
                continue

            for img_name in os.listdir(label_dir):
                if img_name.endswith(("png", "jpg", "jpeg", "bmp", "tiff")):
                    img_path = os.path.join(label_dir, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(label)

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

        return image, label

    def show_random_images(dataset, num_images=16):
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        axes = axes.flatten()

        indices = random.sample(range(len(dataset)), num_images)
        for ax, idx in zip(axes, indices):
            image, label = dataset[idx]
            image = image.permute(1, 2, 0)

            ax.imshow(image)
            ax.set_title(f"Label: {label}")
            ax.axis("off")

        plt.tight_layout()
        plt.show()

    def show_one_image_per_directory(dataset, num_dirs=27):
        fig, axes = plt.subplots(3, 9, figsize=(18, 6))
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
                axes[count].set_title(f"Label: {label}")
                axes[count].axis("off")
                count += 1

        plt.tight_layout()
        plt.show()
