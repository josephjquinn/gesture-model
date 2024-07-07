from torch.utils.data import DataLoader
from torch.utils.data import random_split
from util.dataset import landmarkDataset, imageDataset
import torch


def create_image_dataloaders(
    data_dir: str,
    resize: (int, int),
    normalize: False,
    batch_size: int,
):
    dataset = imageDataset(root_dir=data_dir, resize=resize, normalize=normalize)
    num_data = len(dataset)
    num_train = int(0.8 * num_data)
    num_val = int(0.1 * num_data)
    num_test = num_data - num_train - num_val

    train_data, val_data, test_data = random_split(
        dataset,
        [num_train, num_val, num_test],
        generator=torch.Generator().manual_seed(42),
    )
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=True,
    )
    test_dataloader = DataLoader(
        test_data,
        shuffle=False,
    )

    print("train size: ", train_data.__len__())
    print("val size", val_data.__len__())
    print("test size", test_data.__len__())

    return train_dataloader, val_dataloader, test_dataloader


def create_landmark_dataloaders(
    file: str,
    batch_size: int,
    normalize: bool,
    center_wrist: bool,
):
    dataset = landmarkDataset(file)
    num_data = len(dataset)
    num_train = int(0.8 * num_data)
    num_val = int(0.1 * num_data)
    num_test = num_data - num_train - num_val

    train_data, val_data, test_data = random_split(
        dataset,
        [num_train, num_val, num_test],
        generator=torch.Generator().manual_seed(42),
    )
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=True,
    )
    test_dataloader = DataLoader(
        test_data,
        shuffle=False,
    )

    print("train size: ", train_data.__len__())
    print("val size", val_data.__len__())
    print("test size", test_data.__len__())

    return train_dataloader, val_dataloader, test_dataloader
