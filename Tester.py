from models.Separator import AttnNet
import Utils

import musdb
import torch
import numpy as np
import os


def estimate_track(model, track, input_length, get_betamap=False):
    """
    separate one full track

    :param model:
    :param track: desired track to separate (tensor) : (bs=1, ch=1, t)
    :return: list of estimated sources : [ (1, 1, t), (1, 1, t) .. ]

    """
    track_length = track.shape[2]

    source_preds = [np.zeros(track.shape, np.float32) for _ in range(2)]

    # save beta map
    if get_betamap:
        beta_maps = {s: [[] for _ in range(9)] for s in range(2)}

    for start in range(0, track_length, input_length):
        if start + input_length > track_length:
            start = track_length - input_length - 1

        mix_segment = track[:, :, start: start+input_length]

        # betas : tuple (beta1, beta2), beta1 : list of beta maps for skip connections, (bs, c, t)
        estimates, betas = model(mix_segment)

        # when gpu is on, .cpu() needed
        estimates = [estimate.detach().cpu().numpy() for estimate in estimates]

        for s in range(len(source_preds)):
            source_preds[s][:, :, start: start+input_length] = estimates[s]

            if get_betamap:
                for i, skip in enumerate(betas[s]):
                    beta_maps[s][i].append(skip.detach().cpu().numpy().squeeze())

    if get_betamap:
        return source_preds, beta_maps

    return source_preds


def save_heatmap(attention_map, path):
    """

    :param attention_map: array of attention map, (attention proportion, t)
    :param path: path to save
    :return:
    """
    import seaborn as sns
    import matplotlib.pyplot as plt

    xticklabels = attention_map.shape[1]//5
    ax = sns.heatmap(attention_map, xticklabels=xticklabels, yticklabels=20)
    # plt.show()
    figure = ax.get_figure()
    figure.savefig(path, dpi=400) # 400
    plt.clf()


def save_all_heatmaps(attention_maps, exp_path):
    """

    :param attention_maps: {0: [skip1, skip2, ...] 1: [skip1, skip2, ...]}, each skip (c, t)
    :param exp_path: experiment_path
    :return:
    """
    for s in range(len(attention_maps)):
        for i, skip in enumerate(attention_maps[s]):

            save_path = os.path.join(exp_path, 'source{}_skip{}'.format(s, i))
            save_heatmap(np.squeeze(np.concatenate(skip, axis=1)), save_path)


class Tester:
    def __init__(self, target_wav, config):

        self.attention_fn = config.attention_fn

        self.input_length = config.input_length

        self.num_layers = config.num_layers
        self.num_filters = config.num_filters

        self.enc_filter_size = config.enc_filter_size
        self.dec_filter_size = config.dec_filter_size

        self.AttnNet = AttnNet(attention_fn=self.attention_fn, num_layers=self.num_layers,
                               num_filters=self.num_filters, enc_filter_size=self.enc_filter_size,
                               dec_filter_size=self.dec_filter_size)

        # load target wav
        if target_wav is None:
            self.data_path = config.data_path
            self.load_files()
        else:
            self.sequences = {os.path.basename(target_wav)[:-4]: Utils.load_wav(target_wav)}

        # load model
        self.exp_path = config.exp_path
        self.saved_model = config.saved_model

        self.load_model()

    def load_files(self):
        mus = musdb.DB(root_dir=self.data_path)
        tracks = mus.load_mus_tracks(subsets=['test'])
        tracks = [track.name for track in tracks]

        # list of (mix, accompany, vocal)
        self.sequences = Utils.load_tracks(os.path.join(self.data_path, 'test'), tracks, include_mix=True)
        self.sequences = {track: self.sequences[track] for track in tracks}

    def load_model(self):
        print('Loading the trained models from {}...'.format(self.saved_model))
        model_path = os.path.join(self.exp_path, self.saved_model)
        self.AttnNet.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))

    def predict(self):
        """
        self.sequences : list of triples (mix, accompany, vocal), here each component is array [ch=1, t]

        :return:
        """
        save_folder = os.path.join(self.exp_path, self.saved_model.split('.')[0])

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for track_name, track in self.sequences.items():

            track = torch.from_numpy(track).unsqueeze(0).float()
            source_pred, beta_maps = estimate_track(self.AttnNet, track, self.input_length, get_betamap=True)

            for i, source in enumerate(source_pred):
                source = np.squeeze(source)

                save_path = os.path.join(save_folder, track_name + 'source{}.wav'.format(i))
                Utils.write_wav(source, save_path, 22050)

            # save beta maps
            save_all_heatmaps(beta_maps, save_folder)

if __name__ == '__main__':
    pass