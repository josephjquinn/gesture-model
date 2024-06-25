import torch
from model.cnnet import cnn
import torch.nn as nn
from torch.utils.data import random_split
from util.dataLoader import gestureDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import util.engine

data_path = "./data/"
num_epochs = 100
batch_size = 100
learning_rate = 0.01
TRAIN_SPLIT = 0.75
VAL_SPLIT = 1 - TRAIN_SPLIT

transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

dataset = gestureDataset(root_dir=data_path, transform=transform)

num_data = len(dataset)
num_train = int(0.8 * num_data)
num_val = int(0.1 * num_data)
num_test = num_data - num_train - num_val

train_data, val_data, test_data = random_split(
    dataset, [num_train, num_val, num_test], generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)
test_loader = DataLoader(test_data)

trainSteps = len(train_loader.dataset) // batch_size
valSteps = len(val_loader.dataset) // batch_size


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = cnn().to(device)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

H = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
print("[INFO] training the network...")


util.engine.train(
    model=model,
    train_dataloader=train_loader,
    test_dataloader=test_loader,
    loss_fn=criterion,
    optimizer=optimizer,
    epochs=10,
    device=device,
)
