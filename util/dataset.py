import os
from PIL import Image
import torch
import numpy as np
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


class landmarkDataset(Dataset):
    def __init__(self, pickle_file):
        self.data = self.load_data(pickle_file)

    def __len__(self):
        return len(self.data)

    def __getshape__(self):
        return (self.__len__(), *self.__getitem__(0)[0].shape)

    def __getitem__(self, idx):
        sample = self.data[idx]
        landmarks = sample["landmarks"]
        label = sample["label"]

        landmarks = torch.tensor(landmarks, dtype=torch.float32)
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

    def show_random_landmarks(self, n=5):
        random_samples_idx = random.sample(range(len(self)), k=n)

        fig, axes = plt.subplots(1, n, figsize=(n * 4, 4))

        for i, targ_sample in enumerate(random_samples_idx):
            landmarks, label = self[targ_sample]
            num_landmarks = len(landmarks)
            annotated_image = np.ones((512, 512, 3), dtype=np.uint8) * 255

            # Convert landmarks to the format expected by MediaPipe
            landmark_list = []
            for landmark in landmarks:
                x, y, z = landmark
                landmark_list.append(
                    mp.framework.formats.landmark_pb2.NormalizedLandmark(x=x, y=y, z=z)
                )

            hand_landmarks = mp.framework.formats.landmark_pb2.NormalizedLandmarkList(
                landmark=landmark_list
            )

            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=hand_landmarks,
                connections=mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style(),
            )

            axes[i].imshow(annotated_image)
            axes[i].set_title(f"Label: {label.item()}", fontsize=8)
            axes[i].axis("off")

        plt.tight_layout()
        plt.show()
