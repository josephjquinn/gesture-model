import os
from PIL import Image
import torch
import random
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pickle
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


class imageDataset(Dataset):
    def __init__(self, root_dir, resize=None, normalize=False):
        self.root_dir = root_dir
        self.resize = resize
        self.normalize = normalize
        self.image_paths = []
        self.labels = []

        label_dirs = sorted(os.listdir(root_dir))
        print(label_dirs)

        for label_idx, label_name in enumerate(label_dirs):
            label_dir = os.path.join(root_dir, label_name)
            if not os.path.isdir(label_dir):
                continue

            for img_name in os.listdir(label_dir):
                if img_name.endswith(("png", "jpg", "jpeg", "bmp", "tiff")):
                    img_path = os.path.join(label_dir, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(label_idx)

        if self.normalize:
            self.mean, self.std = self.calculate_mean_std()
        else:
            self.mean, self.std = None, None

        self.transform = self.build_transform()

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

    def build_transform(self):
        transform_list = []

        if self.resize:
            transform_list.append(transforms.Resize((self.resize, self.resize)))

        transform_list.append(transforms.ToTensor())

        if self.normalize:
            transform_list.append(transforms.Normalize(self.mean, self.std))

        return transforms.Compose(transform_list)

    def calculate_mean_std(self):
        mean = torch.zeros(3)
        std = torch.zeros(3)

        num_samples = len(self.image_paths)

        for img_path in self.image_paths:
            img = Image.open(img_path).convert("RGB")
            img_tensor = transforms.ToTensor()(img)
            mean += torch.mean(img_tensor, dim=(1, 2))
            std += torch.std(img_tensor, dim=(1, 2))

        mean /= num_samples
        std /= num_samples

        return mean.tolist(), std.tolist()

    def display_image_from_index(self, idx):
        """
        Displays an image from the dataset at the given index.
        """
        # Retrieve the image and label
        image, label = self[idx]

        # Convert tensor to PIL image
        image = transforms.ToPILImage()(image)

        # Display image
        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        plt.title(f"Label: {label}")
        plt.axis("off")
        plt.show()


class landmarkDataset(Dataset):
    def __init__(self, pickle_file, normalize=True, center_wrist=False):
        self.data = self.load_data(pickle_file)
        self.normalize = normalize
        self.center_wrist = center_wrist

    def __len__(self):
        return len(self.data)

    def __getshape__(self):
        return (self.__len__(), *self.__getitem__(0)[0].shape)

    def __getitem__(self, idx):
        sample = self.data[idx]
        landmarks = sample["landmarks"]
        label = sample["label"]

        landmarks = torch.tensor(landmarks, dtype=torch.float32)

        if self.normalize:
            landmarks = self.normalize_coordinates(landmarks)

        if self.center_wrist:
            landmarks = self.center_coordinates(landmarks)

        label = torch.tensor(label)
        return landmarks, label

    def load_data(self, pickle_file):
        with open(pickle_file, "rb") as f:
            data = pickle.load(f)
        flattened_data = []
        for frame_landmarks in data:
            for sample in frame_landmarks:
                flattened_data.append(sample)
        return flattened_data

    def display_landmarks(self, position):
        landmarks, label = self[position]
        x = landmarks[:, 0]
        y = landmarks[:, 1]
        z = landmarks[:, 2]

        # Plotting 3D
        fig = plt.figure(figsize=(14, 6))
        ax1 = fig.add_subplot(121, projection="3d")
        ax1.scatter(x, y, z, c="green", marker="o", linewidths=6)

        finger_connections = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),  # Thumb
            (0, 5),
            (5, 6),
            (6, 7),
            (7, 8),  # Index finger
            (9, 10),
            (10, 11),
            (11, 12),  # Middle finger
            (13, 14),
            (14, 15),
            (15, 16),  # Ring finger
            (0, 17),
            (17, 18),
            (18, 19),
            (19, 20),  # Little finger
            (0, 5),
            (5, 9),
            (9, 13),
            (13, 17),
        ]

        for connection in finger_connections:
            start, end = connection
            ax1.plot(
                [x[start], x[end]],
                [y[start], y[end]],
                [z[start], z[end]],
                color="black",
                linewidth=5,
            )

        ax1.set_xlabel("X Axis")
        ax1.set_ylabel("Y Axis")
        ax1.set_zlabel("Z Axis")
        ax1.view_init(elev=-70, azim=-90)

        # Plotting 2D
        ax2 = fig.add_subplot(122)
        ax2.scatter(x, y, c="r", marker="o")

        for connection in finger_connections:
            start, end = connection
            ax2.plot([x[start], x[end]], [y[start], y[end]], color="r")

        ax2.set_xlabel("X Axis")
        ax2.set_ylabel("Y Axis")
        ax2.set_ylim(ax2.get_ylim()[::-1])
        ax2.set_title(f"Label: {label}")

        plt.tight_layout()
        plt.show()

    def normalize_coordinates(self, coords):
        min_val = coords.min(0)[0]
        max_val = coords.max(0)[0]
        normalized_coords = (coords - min_val) / (max_val - min_val)
        return normalized_coords

    def center_coordinates(self, coords):
        wrist_offset = coords[0]  # Assuming wrist is at index 0
        centered_coords = coords - wrist_offset
        return centered_coords
