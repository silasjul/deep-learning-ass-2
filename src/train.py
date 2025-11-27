import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import torch
from medmnist import Evaluator
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
        for inputs, targets in tqdm(dataset.train_loader):
            # forward + backward + optimize
            optimizer.zero_grad()
            outputs = model(inputs)

            targets = targets.squeeze().long()
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

        print("==> Evaluating ...")
        test(model, "train")
        test(model, "test")


def test(model: Model, split: str):
    model.eval()
    y_true = torch.tensor([])
    y_score = torch.tensor([])

    data_loader = model.train_loader if split == "train" else model.test_loader

    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs)

            targets = targets.squeeze().long()
            outputs = outputs.softmax(dim=-1)
            targets = targets.float().resize_(len(targets), 1)

            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)

        y_true = y_true.numpy()
        y_score = y_score.detach().numpy()

        evaluator = Evaluator("pathmnist", split)
        metrics = evaluator.evaluate(y_score)

        print("%s  auc: %.3f  acc:%.3f" % (split, *metrics))
