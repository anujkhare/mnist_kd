import math
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import dataloader
from typing import Tuple
import warnings


def split_train_val(df_train_val, val_frac=0.1, shuffle=True):
    np.random.seed(1123)
    if shuffle:
        df_train_val = df_train_val.sample(frac=1)

    n_train = math.ceil((1-val_frac) * df_train_val.shape[0])

    df_train = df_train_val.iloc[:n_train]
    df_val = df_train_val.iloc[n_train:]

    assert df_train.shape[0] + df_val.shape[0] == df_train_val.shape[0]
    return df_train, df_val


def get_dataset(df) -> Tuple[np.array, np.array]:
    if 'label' not in df:
        warnings.warn('This dataframe does not have labels! Using default label = -100')
        df = df.copy()
        df['label'] = -100

    # Extract numpy arrays
    labels = df['label'].values
    images = df.drop('label', axis=1).values
    assert len(labels) == len(images)

    # Reshape the images to make 28*28
    images = images.reshape(-1, 1, 28, 28)

    # Pad to 32*32
    images = np.pad(images, ((0, 0), (0, 0), (2, 2), (2, 2)), mode='constant', constant_values=0)

    # Convert to pytorch Tensors
    images = torch.FloatTensor(images.astype(np.float32))
    labels = torch.LongTensor(labels.astype(np.int))

    # Create a dataset object
    dataset = torch.utils.data.TensorDataset(images, labels)
    assert len(dataset) == len(df)
    return dataset


def visualize_batch(batch):
    images, labels = batch

    batch_size = len(images)
    plt.figure(figsize=(20, 10))

    for ix in range(batch_size):
        plt.subplot(1, batch_size, ix + 1)
        plt.imshow(images[ix].squeeze(), cmap='gray')
        plt.title(labels[ix].data.numpy())


def visualize_data(dataloader, n=4):
    for ix, next_batch in enumerate(dataloader):
        if ix >= n:
            break
        visualize_batch(next_batch)
