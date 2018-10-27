import musdb
import torch
from torch.utils import data

import random
import numpy as np
import os

import Utils

class Dataset(data.Dataset):

    def __init__(self, dataset_path, tracks, input_length, is_augment=True):

        self.dataset_path = os.path.join(dataset_path, 'train')
        self.tracks = tracks
        self.input_length = input_length

        self.is_augment = is_augment
        self.load_files()

    def load_files(self):
        """
        sequences : list of tuples (accompanient, vocal)

        :return:
        """
        self.sequences = Utils.load_tracks(self.dataset_path, self.tracks)

    def __getitem__(self, track):
        """
        :param track: track number
        :return:
        """
        accompany, vocal = self.sequences[track]
        length = accompany.shape[0]

        start_idx = np.squeeze(np.random.randint(0, length - self.input_length +1, 1))
        end_idx = start_idx + self.input_length

        accompany = accompany[start_idx:end_idx]
        vocal = vocal[start_idx:end_idx]

        if self.is_augment:
            accompany *= np.random.uniform(0.7, 1.0)
            vocal *= np.random.uniform(0.7, 1.0)

        mix = accompany + vocal

        return mix, accompany, vocal
        # return accompany[start_idx:end_idx], vocal[start_idx:end_idx]


    def __len__(self):
        """Return the number of tracks."""
        return len(self.sequences)

def get_loader(dataset_path, input_length, batch_size, num_workers=2):

    # to fix valid set and training result
    np.random.seed(1337)

    mus = musdb.DB(root_dir=dataset_path)
    tracks = mus.load_mus_tracks(subsets=['train'])
    tracks = [track.name for track in tracks]

    np.random.shuffle(tracks)

    train_tracks = tracks[:75]
    valid_tracks = tracks[75:]

    train_dataset = Dataset(dataset_path, train_tracks, input_length)
    valid_dataset = Dataset(dataset_path, valid_tracks, input_length)

    train_loader = data.DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers)

    valid_loader = data.DataLoader(dataset=valid_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers)

    return train_loader, valid_loader


if __name__ == '__main__':
    # test
    train_loader, _ = get_loader(
        dataset_path='../dataset/musdb18',
        input_length=16384,
        batch_size=16
    )

    data_iter = iter(train_loader)

    mix, accompanies, vocals = next(data_iter)

    print(mix.shape, accompanies.shape, vocals.shape)
    print(vocals[0])