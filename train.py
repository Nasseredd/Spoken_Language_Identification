from audio_dataset import AudioDataset

from torch.utils.data import Subset, random_split, DataLoader

import torch
import cnn as our_cnn
import os


def read_dataset(root_dir: str = 'Data', sampleSize: int = 1000, testRep: float = 0.2) -> tuple[Subset]:
    """Create Sample instances for the training and test subsets. 
    It loads all files from <root_dir>/train and <root_dir>/test, then sample them according to <sampleSize> and <testRep>.
    Will hopefully contain the validation set, in the future.

    Args:
        root_dir (str, optional): The dataset's root directory. Needs to contain "train" and "test". Defaults to 'Data'.
        sampleSize (int, optional): The size of the training set sample (training subset). Defaults to 1000.
        testRep (float, optional): The repartition of the test set sample compared i.r.t. the training one (sampleSize * testRep). Defaults to 0.2.

    Returns:
        tuple[Subset]: Two Subset instances, corresponding to the training and test subsets.
    """
    # Create paths
    train_path = os.path.join(".", root_dir, "train")
    test_path = os.path.join(".", root_dir, "test")

    # Create datasets, then subsets of sampleSize & sampleSize * testRep
    train_set, test_set = AudioDataset(train_path), AudioDataset(test_path)
    train_subset = Subset(train_set, torch.arange(sampleSize))
    test_subset = Subset(test_set, torch.arange(sampleSize * testRep))

    return train_subset, test_subset


def init_dataloaders(*subsets: tuple[Subset]) -> list[DataLoader]:
    return [DataLoader(subset, shuffle=True, batch_size=our_cnn.BATCH_SIZE) for subset in subsets]


def init_and_train_model(train_dataloader: DataLoader):
    model = our_cnn.ConvNet(our_cnn.NUM_CLASSES)

    model.train(train_dataloader)
    torch.save(model.state_dict(), f'model.ckpt')


def test_model(test_dataloader: DataLoader):
    model = torch.load("model.ckpt")

    model.test(test_dataloader)


def main(dir: str = 'Audios_tmp', sample_train: float = 0.8):
    samples = read_dataset(dir, sample_train)
    train_dl, test_dl = init_dataloaders(*samples)
    init_and_train_model(train_dl)
    test_model(test_dl)


if __name__ == "__main__":
    main()
