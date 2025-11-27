from medmnist import PathMNIST, INFO
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from .utils.config import BATCH_SIZE

info = INFO["pathmnist"]
task = info["task"]
n_channels = info["n_channels"]
n_classes = len(info["label"])

data_transform = transforms.Compose(
    [  # mean and std found in github documentation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
)

train_dataset = PathMNIST(split="train", transform=data_transform, download=True)
val_dataset = PathMNIST(split="val", transform=data_transform, download=True)
test_dataset = PathMNIST(split="test", transform=data_transform, download=True)


class Dataset:
    def __init__(self):
        self.train_loader = data.DataLoader(
            dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True
        )
        self.val_loader = data.DataLoader(
            dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False
        )
        self.test_loader = data.DataLoader(
            dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False
        )

    def print_dataset_info(self):
        print("task:", task)
        print("number of channels:", n_channels)
        print("number of classes:", n_classes)

        print("\ntrain_dataset.shape:", train_dataset.imgs.shape)
        print("val_dataset.shape:", val_dataset.imgs.shape)
        print("test_dataset.shape:", test_dataset.imgs.shape)

    def visualize_data(self):
        samples_per_class = 5
        fig, axes = plt.subplots(
            n_classes, samples_per_class, figsize=(samples_per_class * 2, n_classes * 2)
        )
        imgs = train_dataset.imgs
        labels = train_dataset.labels.flatten()
        for class_idx in range(n_classes):
            class_indices = np.where(labels == class_idx)[0][:samples_per_class]
            for i, idx in enumerate(class_indices):
                img = imgs[idx]
                axes[class_idx, i].imshow(img)
                axes[class_idx, i].set_title(f"Class {class_idx}")
                axes[class_idx, i].axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    dataset = Dataset()
    dataset.print_dataset_info()
    dataset.visualize_data()
