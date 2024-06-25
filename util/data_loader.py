from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from util.dataset import gestureDataset
import torch


def create_dataloaders(
    data_dir: str,
    transform: transforms.Compose,
    batch_size: int,
):
    dataset = gestureDataset(root_dir=data_dir, transform=transform)
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
