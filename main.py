import os

import torch
from torch.utils.data import DataLoader, Subset, random_split

import cnn as our_cnn
from audio_dataset import AudioDataset

# Default path to the dataset
DEFAULT_DATA_PATH = os.path.normpath('C:\\Users\\user\\Desktop\\train')

# Hyper-parameters: number of epochs, batch size, learning rate, default sample size (train. set)
NUM_EPOCHS = 10
BATCH_SIZE = 100
LEARNING_RATE = 0.002
DEFAULT_SAMPLE_SIZE = 5000

def read_dataset(root_dir: str, sampleSize: int, testRep: float = 0.15, valRep: float = 0.15) -> tuple[Subset]:
    """Create Sample instances for the training, validation and test subsets. 
    It loads all files from <root_dir>/train and <root_dir>/test, then sample them according to <sampleSize>, <testRep>,  and <valRep>.

    Args:
        root_dir (str): The dataset's root directory. Needs to contain "train" and "test".
        sampleSize (int): The size of the training set sample (training subset).
        testRep (float, optional): The repartition of the test set sample compared i.r.t. the training one (sampleSize * testRep). Defaults to 0.15.
        valRep (float, optional): The repartition of the validation set sample compared i.r.t. the training one. Defaults to 0.15.

    Returns:
        tuple[Subset]: Three Subset instances, corresponding to the training, validation, and test subsets.
    """
    # Create paths
    train_path = os.path.join(".", root_dir, "train")
    test_path = os.path.join(".", root_dir, "test")

    # Create datasets, then subsets of sampleSize, sampleSize * valRep, and sampleSize * testRep
    train_set, test_set = AudioDataset(train_path), AudioDataset(test_path)

    # Train subsets (train, val, and reminder)
    val_size = int(sampleSize * valRep)
    reminder = len(train_set) - val_size - sampleSize
    train_subset, val_subset, _ = random_split(train_set, [sampleSize, val_size, reminder])

    # Test subset (test, and reminder)
    test_size = int(sampleSize * testRep)
    reminder = len(test_set) - test_size
    test_subset, _ = random_split(test_set, [test_size, reminder])

    assert 0 not in (len(train_subset), len(val_subset), len(test_subset)), "At least one of the subsets is empty. Please check the paths."
    return train_subset, val_subset, test_subset


def init_dataloaders(*subsets: tuple[Subset]) -> list[DataLoader]:
    """Initialize the dataloaders.
    Initialize the dataloader with the Subset/Dataset objects added in argument. 
    Shuffle and set the batch size to the hyper-parameter defined in cnn.py.

    Args:
        subsets (tuple[Subset], (technically) optional): a tuple of subsets to create dataloaders from.

    Returns:
        list[DataLoader]: A list of DataLoader instances, corresponding to each Subset.
    """
    return [DataLoader(subset, shuffle=True, batch_size=BATCH_SIZE) for subset in subsets]


def init_and_train_model(train_dataloader: DataLoader, validation_dataloader: DataLoader, o_file: str = 'model.ckpt'):
    """Initialize and train a CNN model.
    It creates a CNN object, then train it using a training and a validation dataloaders.
    Then, save the model's parameters into o_file.

    Args:
        train_dataloader (DataLoader): DataLoader for the training set.
        validation_dataloader (DataLoader):  DataLoader for the validation set.
        o_file (str, optional): Path to the file we will write the model parameters into. Defaults to 'model.ckpt'.
    """
    model = our_cnn.ConvNet(NUM_EPOCHS, LEARNING_RATE)

    model.train_model(train_dataloader, validation_dataloader)
    torch.save(model.state_dict(), o_file)


def test_model(test_dataloader: DataLoader, parameter_file: str = 'model.ckpt'):
    """Load and test the model.
    Load the model from a file containing its parameters, then execute its testing session (test()).

    Args:
        test_dataloader (DataLoader): A DataLoader instance containing the testing data.
        parameter_file (str, optional): Path to the file containing the model parameters. Defaults to 'model.ckpt'.
    """
    model = our_cnn.ConvNet(NUM_EPOCHS, LEARNING_RATE)
    model.load_state_dict(torch.load(parameter_file))

    model.test(test_dataloader)


def main(root_dir: str, sampleSize: int, testRep: float, valRep: float, parameter_file: str):
    """Main function.
    Executes this file's function.

    Args:
        root_dir (str): The dataset's root directory. Needs to contain "train" and "test".
        sampleSize (int): The size of the training set sample (training subset).
        testRep (float): The repartition of the test set sample compared i.r.t. the training one (sampleSize * testRep).
        valRep (float): The repartition of the validation set sample compared i.r.t. the training one.
        parameter_file (str): Path to the file that will contain the model parameters.
    """
    samples = read_dataset(root_dir, sampleSize, testRep, valRep)
    train_dl, val_dl, test_dl = init_dataloaders(*samples)
    init_and_train_model(train_dl, val_dl, parameter_file)
    test_model(test_dl, parameter_file)


if __name__ == "__main__":
    main(root_dir=DEFAULT_DATA_PATH, sampleSize=DEFAULT_SAMPLE_SIZE,
         testRep=0.15, valRep=0.15, parameter_file='model.ckpt')
