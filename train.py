import os

import torch
from torch.utils.data import DataLoader, Subset, random_split

import cnn as our_cnn
from audio_dataset import AudioDataset


def read_dataset(root_dir: str = 'Data', sampleSize: int = 1000, testRep: float = 0.15, valRep: float = 0.15) -> tuple[Subset]:
    """Create Sample instances for the training, validation and test subsets. 
    It loads all files from <root_dir>/train and <root_dir>/test, then sample them according to <sampleSize>, <testRep>,  and <valRep>.

    Args:
        root_dir (str, optional): The dataset's root directory. Needs to contain "train" and "test". Defaults to 'Data'.
        sampleSize (int, optional): The size of the training set sample (training subset). Defaults to 1000.
        testRep (float, optional): The repartition of the test set sample compared i.r.t. the training one (sampleSize * testRep). Defaults to 0.15.
        valRep (float, optional): The repartition of the validation set sample compared i.r.t. the training one. Defaults to 0.15.

    Returns:
        tuple[Subset]: Two Subset instances, corresponding to the training and test subsets.
    """
    # Create paths
    train_path = os.path.join(".", root_dir, "train")
    test_path = os.path.join(".", root_dir, "test")

    # Create datasets, then subsets of sampleSize, sampleSize * valRep, and sampleSize * testRep
    train_set, test_set = AudioDataset(train_path), AudioDataset(test_path)
    train_subset = Subset(train_set, torch.arange(sampleSize))
    train_subset, val_subset = random_split(train_subset, [1 - valRep, valRep])

    test_subset = Subset(test_set, torch.arange(sampleSize * testRep))

    return train_subset, val_subset, test_subset


def init_dataloaders(*subsets: tuple[Subset]) -> list[DataLoader]:
    return [DataLoader(subset, shuffle=True, batch_size=our_cnn.BATCH_SIZE) for subset in subsets]


def init_and_train_model(train_dataloader: DataLoader, validation_dataloader: DataLoader):
    model = our_cnn.ConvNet(our_cnn.NUM_CLASSES)

    model.train(train_dataloader, validation_dataloader)
    torch.save(model.state_dict(), f'model.ckpt')


def test_model(test_dataloader: DataLoader):
    model = torch.load("model.ckpt")

    model.test(test_dataloader)


def main(root_dir: str = '/c/Users/user/Desktop/', sampleSize: int = 1000, testRep: float = 0.15, valRep: float = 0.15):
    samples = read_dataset(root_dir, sampleSize, testRep, valRep)
    train_dl, val_dl, test_dl = init_dataloaders(*samples)
    init_and_train_model(train_dl, val_dl)
    test_model(test_dl)


if __name__ == "__main__":
    main()
