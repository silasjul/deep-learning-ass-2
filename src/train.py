import torch.optim as optim
import torch.nn as nn
import tqdm
from .model import Model
from .dataset import Dataset, in_channels, num_classes
from .utils.config import EPOCHS, lr


def train():
    dataset = Dataset()
    model = Model(in_channels, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    for epoch in range(EPOCHS):
        train_correct = 0
        train_total = 0
        test_correct = 0
        test_total = 0

        model.train()
        for inputs, targets in tqdm(dataset.train_loader):
            # forward + backward + optimize
            optimizer.zero_grad()
            outputs = model(inputs)

            targets = targets.squeeze().long()
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()
