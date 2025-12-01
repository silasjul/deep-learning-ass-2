import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
from .dataset import test_dataset, data_transform
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from .model import Model
from .dataset import Dataset, n_channels, n_classes
from .utils.config import EPOCHS, lr


def train():
    dataset = Dataset()
    model = Model(n_channels, n_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    train_accs = []
    val_accs = []

    for epoch in range(EPOCHS):
        model.train()
        for inputs, targets in tqdm(dataset.train_loader, desc=f"Epoch {epoch+1}"):
            # forward + backward + optimize
            optimizer.zero_grad()
            outputs = model(inputs)

            targets = targets.squeeze().long()
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

        print("==> Evaluating ...")
        train_acc = evaluate(model, dataset, "train")
        val_acc = evaluate(model, dataset, "val")
        train_accs.append(train_acc)
        val_accs.append(val_acc)

    # Plot accuracy over EPOCHS
    epochs = list(range(1, EPOCHS + 1))
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accs, label="Train Accuracy")
    plt.plot(epochs, val_accs, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()

    test(model, dataset)


def evaluate(model: Model, dataset: Dataset, split: str):
    model.eval()
    y_true = torch.tensor([], dtype=torch.long)
    y_score = torch.tensor([])
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0

    if split == "train":
        data_loader = dataset.train_loader
    elif split == "val":
        data_loader = dataset.val_loader
    else:
        data_loader = dataset.test_loader

    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs)

            targets = targets.squeeze().long()
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            outputs = outputs.softmax(dim=-1)

            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)

        y_true = y_true.numpy()
        y_score = y_score.detach().numpy()

        acc = accuracy_score(y_true, y_score.argmax(axis=1))
        auc = roc_auc_score(y_true, y_score, multi_class="ovr", average="macro")

        print(f"{split} auc: {auc:.4f}  acc: {acc:.4f}")

        return acc


def test(model, dataset):
    model.eval()

    # Compute predictions for all test data for confusion matrix
    all_preds = []
    all_actuals = []
    y_score = torch.tensor([])
    with torch.no_grad():
        for inputs, targets in dataset.test_loader:
            outputs = model(inputs)
            outputs = outputs.softmax(dim=-1)
            preds = outputs.argmax(dim=1).cpu().numpy()
            actuals = targets.squeeze().cpu().numpy()
            all_preds.extend(preds)
            all_actuals.extend(actuals)
            y_score = torch.cat((y_score, outputs.cpu()), 0)

    # Compute confusion matrix
    cm = confusion_matrix(all_actuals, all_preds, labels=range(n_classes))

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(n_classes)
    plt.xticks(tick_marks, range(n_classes))
    plt.yticks(tick_marks, range(n_classes))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    plt.show()

    # Compute test metrics
    accuracy = accuracy_score(all_actuals, all_preds)
    auc = roc_auc_score(
        all_actuals, y_score.numpy(), multi_class="ovr", average="macro"
    )
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_actuals, all_preds, average="macro"
    )

    # Sanity check - sample 5 random images per class for plotting
    imgs = test_dataset.imgs
    labels = test_dataset.labels.flatten()
    sampled_per_class = [[] for _ in range(n_classes)]
    for class_idx in range(n_classes):
        class_indices = np.where(labels == class_idx)[0]
        n_samples = min(8, len(class_indices))
        sampled_indices = np.random.choice(class_indices, n_samples, replace=False)
        for idx in sampled_indices:
            img = imgs[idx]
            img_tensor = data_transform(img).unsqueeze(0)
            with torch.no_grad():
                output = model(img_tensor)
                pred = output.argmax(dim=1).item()
            actual = labels[idx]
            sampled_per_class[class_idx].append((img, pred, actual))

    # Plot sampled images
    fig, axes = plt.subplots(n_classes, 8, figsize=(10, 2 * n_classes))
    for class_idx in range(n_classes):
        for j in range(5):
            if j < len(sampled_per_class[class_idx]):
                img, pred, actual = sampled_per_class[class_idx][j]
                axes[class_idx, j].imshow(img)
                axes[class_idx, j].set_title(f"Pred: {pred}, Actual: {actual}")
            axes[class_idx, j].axis("off")
    plt.tight_layout()
    plt.show()

    # Print test evaluation metrics
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test AUC: {auc:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
