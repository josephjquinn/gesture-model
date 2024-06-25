import torch
from model.cnnet import cnn
import torch.nn as nn
from torchvision import transforms
import util.engine
import util.data_loader

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


train_loader, val_loaader, test_loader = util.data_loader.create_dataloaders(
    "./data/", transform, 100
)


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
