import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from .model import Model
from .dataset import Dataset, n_channels, n_classes
from .utils.config import EPOCHS, lr


def train():
    dataset = Dataset()
    model = Model(n_channels, n_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

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
        test(model, dataset, "train")
        test(model, dataset, "test")


def test(model: Model, dataset: Dataset, split: str):
    model.eval()
    y_true = torch.tensor([], dtype=torch.long)
    y_score = torch.tensor([])

    data_loader = dataset.train_loader if split == "train" else dataset.test_loader

    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs)

            targets = targets.squeeze().long()
            outputs = outputs.softmax(dim=-1)

            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)

        y_true = y_true.numpy()
        y_score = y_score.detach().numpy()

        acc = accuracy_score(y_true, y_score.argmax(axis=1))
        auc = roc_auc_score(y_true, y_score, multi_class="ovr", average="macro")

        print(f"{split} auc: {auc:.4f}  acc: {acc:.4f}")
