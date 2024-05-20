import argparse
import datetime
from math import log
import os
import mlflow
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from mlflow import log_metric, log_figure
from mlflow.pytorch import log_model
from mnist_classifier import MNISTClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import plotly.express as px


def evaluate(model, dataloader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy


def multiclass_auc(y_true, y_pred, classes):
    n_classes = len(classes)
    y_true_binarized = label_binarize(y_true, classes=classes)
    y_pred_binarized = label_binarize(y_pred, classes=classes)
    roc_curves = []
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_binarized[:, i], y_pred_binarized[:, i])
        roc_auc = auc(fpr, tpr)
        class_label = classes[i]
        roc_curves.append((fpr, tpr, roc_auc, class_label))

    return roc_curves


def main(params, args):

    mlflow.set_tag("mlflow.runName", args.run_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_dataset = MNIST(root="./data", train=False, transform=ToTensor(), download=True)
    val_loader = DataLoader(val_dataset, batch_size=params["batch_size"])

    model = MNISTClassifier(
        params["input_size"], params["hidden_size"], params["num_classes"]
    ).to(device)
    model.load_state_dict(torch.load(args.path, map_location=device))

    log_model(model, "model")

    criterion = nn.CrossEntropyLoss()

    val_accuracy = evaluate(model, val_loader, criterion, device)
    log_metric("val_accuracy", val_accuracy)

    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(torch.argmax(outputs, dim=1).cpu().numpy())

    roc_curves = multiclass_auc(y_true, y_pred, classes=range(params["num_classes"]))

    for fpr, tpr, roc_auc, class_label in roc_curves:
        log_metric(f"{class_label}_auc", roc_auc)
        fig = px.line(x=fpr, y=tpr, title=f"ROC curve for class {class_label}")

        fig.write_html(f"{class_label}_roc_curve.html")
        log_figure(fig, f"{class_label}_roc_curve.html")
        os.remove(f"{class_label}_roc_curve.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        default="mnist_classifier.pth",
        help="Path to model state dict",
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
            f"mnist_validation_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
        )

    params = {
        "input_size": 784,
        "hidden_size": 128,
        "num_classes": 10,
        "batch_size": 64,
    }
    main(params, args)
