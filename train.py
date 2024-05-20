from ast import arg
from datetime import date
import datetime
from typing import Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import mlflow
from mlflow import log_metric, log_param
from mlflow.pytorch import log_model
from mnist_classifier import MNISTClassifier
import argparse
from validate import evaluate


# Training function
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    epoch_loss = running_loss / len(dataloader)
    return epoch_loss


def main(params, args):

    mlflow.set_tag("mlflow.runName", args.run_name)

    # Set device
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

    # Load data
    train_dataset = MNIST(
        root="./data", train=True, transform=ToTensor(), download=True
    )
    train_loader = DataLoader(
        train_dataset, batch_size=params["batch_size"], shuffle=True
    )

    # Create model
    model = MNISTClassifier(
        params["input_size"], params["hidden_size"], params["num_classes"]
    ).to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])

    # Log parameters
    for key, value in params.items():
        log_param(key, value)

    # Train
    for epoch in range(params["num_epochs"]):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        log_metric("train_loss", train_loss, step=epoch)
        print(f"Loss: {train_loss}, Epoch: {epoch}")

    log_model(model, "model")

    # Evaluate
    val_dataset = MNIST(root="./data", train=False, transform=ToTensor(), download=True)
    val_loader = DataLoader(val_dataset, batch_size=params["batch_size"])
    val_accuracy = evaluate(model, val_loader, criterion, device)
    log_metric("val_accuracy", val_accuracy)

    # Save model state dict
    torch.save(
        model.state_dict(),
        args.path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MNIST Classifier")

    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="Number of epochs to train"
    )
    parser.add_argument(
        "--path",
        type=str,
        default="mnist_classifier.pth",
        help="Path to save model state dict",
    )

    parser.add_argument(
        "--run_name",
        type=str,
        default="",
        help="Run name",
    )

    args = parser.parse_args()

    if args.run_name == "":
        args.run_name = (
            f"mnist_train_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
        )

    params = {
        "input_size": 784,
        "hidden_size": 128,
        "num_classes": 10,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "num_epochs": args.num_epochs,
    }
    main(params, args)
