from audio_dataset import AudioDataset

from torch.utils.data import Subset, random_split, DataLoader

import torch
import cnn as our_cnn

def split_dataset(dir: str = 'Audios_tmp', sample_train: float = 0.8, nbElements: int = 1000) -> list[Subset]:
    """Create an AudioDataset instance from a given directory, and split it into two subsets.
    Intended to create training and test sets. Hopefully, a version with a validation set in the future.

    Args:
        dir (str, optional): The path to the dataset repository. Defaults to 'Audios_tmp'.
        sample_train (float, optional): The sample size for the first subset (assumed to be the training one). Defaults to 0.8.
        nbElements:

    Returns:
        list[Subset]: A list of Subset objects, which correspond to the different subsets, split according to the sampling size given in arguments. 

    SeeAlso: 
        torch.utils.data.random_split
    """    

    dataset = AudioDataset(dir)
    dataset = Subset(dataset, torch.arange(nbElements))
    sample_test = 1 - sample_train

    return random_split(dataset, [sample_train, sample_test])

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
    samples = split_dataset(dir, sample_train)
    train_dl, test_dl = init_dataloaders(*samples)
    init_and_train_model(train_dl)
    test_model(test_dl)

if __name__ == "__main__":
    main()