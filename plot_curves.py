import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_losses(path: str = "training_losses.csv", graph_name: str = "Training and validation losses"):
    loss_df = pd.read_csv(path)

    plt.figure()

    t = np.arange(loss_df.shape[0])
    plt.plot(t, loss_df["Training"], label='Training Loss')
    plt.plot(t, loss_df["Validation"], label='Validation Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # plt.axis([1, 20, 1, 6])

    plt.title(graph_name)
    return plt.legend()
