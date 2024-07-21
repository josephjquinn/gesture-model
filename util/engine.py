import torch
from typing import Dict, List, Tuple
from matplotlib import pyplot as plt


def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()

    train_loss, train_acc = 0, 0
    stepcount = 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        stepcount = stepcount + 1

        print(
            f"Batch [{batch}/{len(dataloader)}], X.shape: {list(X.shape)}, y.shape: {list(y.shape)}"
        )

        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y)
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def test_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    test_loss, test_acc = 0, 0
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            test_pred_logits = model(X)
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += (test_pred_labels == y).sum().item() / len(y)
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc


def plot(results: Dict[str, List[float]]) -> None:
    plt.clf()

    plt.subplot(1, 2, 1)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(results["train_loss"], "-r", label="train")
    plt.plot(results["test_loss"], "-b", label="val")
    plt.legend()
    plt.ylim(ymin=0)

    plt.subplot(1, 2, 2)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.plot(results["train_acc"], "-r", label="train")
    plt.plot(results["test_acc"], "-b", label="val")
    plt.legend()
    plt.ylim(0, 1)

    plt.show()
    plt.pause(0.1)


def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epochs: int,
    device: torch.device,
    save_path: str,
) -> Dict[str, List]:
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    plt.ion()
    plt.figure(figsize=(8, 4))

    for epoch in range(epochs):
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )
        val_loss, val_acc = test_step(
            model=model, dataloader=val_dataloader, loss_fn=loss_fn, device=device
        )

        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {val_loss:.4f} | "
            f"test_acc: {val_acc:.4f}"
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(val_loss)
        results["test_acc"].append(val_acc)

        plot(results)
    plt.ioff()
    plt.show()
    torch.save(model.state_dict(), save_path)
    _, test_acc = test_step(
        model=model, dataloader=val_dataloader, loss_fn=loss_fn, device=device
    )
    print(test_acc)
    print(f"Model saved to {save_path}")

    return results
