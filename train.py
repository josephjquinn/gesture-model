import yaml
import torch
import torch.nn as nn
import util.engine
import util.data_loader
from model.cnn import ImageCNN, LandmarkCNN
from model.ann import ImageANN, LandmarkANN
from model.vit import ImageViT
import argparse


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main(config_path):
    # Load YAML config
    config = load_config(config_path)

    # Access values from config
    data_path = config["data"]["data_path"]
    data_type = config["data"]["data_type"]
    batch_size = config["data"]["batch_size"]
    device = config["device"]
    model_name = config["hyper"]["model"]
    epochs = config["hyper"]["epochs"]
    save_path = config["save_path"]
    normalize = config["data"]["normalize"]
    center_wrist = config["data"]["center_wrist"]
    learning_rate = config["hyper"]["learning_rate"]
    optimizer_name = config["hyper"]["optimizer"]
    criterion_name = config["hyper"]["criterion"]
    image_size = config["data"]["image_size"]
    num_classes = config["data"]["num_classes"]

    if data_type == "landmark":
        train_loader, val_loader, test_loader = (
            util.data_loader.create_landmark_dataloaders(
                data_path, batch_size, normalize, center_wrist
            )
        )

    if data_type == "image":
        train_loader, val_loader, test_loader = (
            util.data_loader.create_image_dataloaders(
                data_path, image_size, normalize, batch_size
            )
        )
    print(model_name)
    print(data_type)

    if model_name == "ann" and data_type == "image":
        model = ImageANN(image_size, num_classes).to(device)
    elif model_name == "ann" and data_type == "landmark":
        model = LandmarkANN(num_classes).to(device)
    elif model_name == "cnn" and data_type == "image":
        model = ImageCNN(num_classes).to(device)
    elif model_name == "cnn" and data_type == "landmark":
        model = LandmarkCNN(num_classes).to(device)
    elif model_name == "vit":
        model = ImageViT().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    else:
        raise ValueError("Model not found in config.")

    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Optimizer {optimizer_name} not supported.")

    if criterion_name == "CrossEntropyLoss":
        criterion = nn.CrossEntropyLoss()
    elif criterion_name == "MSELoss":
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Criterion {criterion_name} not supported.")

    print("[INFO] Training the network...")
    res = util.engine.train(
        model=model,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        val_dataloader=val_loader,
        loss_fn=criterion,
        optimizer=optimizer,
        epochs=epochs,
        device=device,
        save_path=save_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model using YAML configuration."
    )
    parser.add_argument("--config", type=str, help="Path to YAML configuration file")

    args = parser.parse_args()

    if args.config:
        config_path = args.config
        main(config_path)
    else:
        print("Please specify a configuration file using --config flag.")
