from src.train import train
from src.dataset import Dataset


def main():
    ds = Dataset()
    ds.visualize_data()
    train()


if __name__ == "__main__":
    main()
