import musdb
from torch.utils import data
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
        self.sequences = Utils.load_tracks(self.dataset_path, self.tracks, include_mix=True)

    def __getitem__(self, track):
        """
        :param track: track number

        accompany, vocal : (c=1, t)
        :return:
        """
        mix, accompany, vocal = self.sequences[track]
        length = accompany.shape[1]

        start_idx = np.squeeze(np.random.randint(0, length - self.input_length +1, 1))
        end_idx = start_idx + self.input_length

        mix = mix[:, start_idx:end_idx]
        accompany = accompany[:, start_idx:end_idx]
        vocal = vocal[:, start_idx:end_idx]

        if self.is_augment:
            accompany *= np.random.uniform(0.7, 1.0)
            vocal *= np.random.uniform(0.7, 1.0)
            mix = accompany + vocal

        return mix, accompany, vocal

    def __len__(self):
        """Return the number of tracks."""
        return len(self.sequences)


def get_loader(dataset_path, input_length, batch_size, num_workers=2):

    # to fix valid set
    np.random.seed(1337)

    def worker_init_fn(worker_id):
        """
        https://github.com/pytorch/pytorch/issues/5059
        to make np.random.randint in __getitem__ operated randomly, we need below function.
        """
        np.random.seed(np.random.get_state()[1][0] + worker_id)

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
                                   num_workers=num_workers,
                                   shuffle=False,
                                   worker_init_fn=worker_init_fn)

    valid_loader = data.DataLoader(dataset=valid_dataset,
                                   batch_size=batch_size,
                                   num_workers=num_workers,
                                   shuffle=False,
                                   worker_init_fn=worker_init_fn)

    return train_loader, valid_loader


if __name__ == '__main__':
    """
    https://github.com/pytorch/pytorch/issues/5059
    
    Make sure that random in __getitem__ works
    
    """
    train_loader, _ = get_loader(
        dataset_path='../dataset/musdb18',
        input_length=16384,
        batch_size=2
    )

    for epoch in range(8):
        np.random.seed()
        print('epoch : {}'.format(epoch))
        for (mix, accompany, vocal) in train_loader:
            print(mix)