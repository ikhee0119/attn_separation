from models.Separator import AttnNet
import Utils

import musdb
import museval

import torch
import torch.nn.functional as f
import numpy as np
import os

import glob
import json


def estimate_track(model, track, input_length, output_length, get_betamap=False, context=True):
    """
    separate one full track

    :param model:
    :param track: desired track to separate (tensor) : (bs=1, ch=1, t) or track object
    :return: list of estimated sources : [ (1, 1, t), (1, 1, t) .. ]

    """
    track_length = track.shape[2]

    source_preds = [np.zeros(track.shape, np.float32) for _ in range(2)]

    # save beta map
    if get_betamap:
        beta_maps = {s: [[] for _ in range(9)] for s in range(2)}

    pad = int((input_length - output_length)/2)
    track = f.pad(track, (pad,pad), 'constant', 0)

    for start in range(0, track_length, output_length):
        if start + output_length > track_length:
            start = track_length - input_length - 1

        mix_segment = track[:, :, start: start+input_length]

        # betas : tuple (beta1, beta2), beta1 : list of beta maps for skip connections, (bs, c, t)
        estimates, betas = model(mix_segment)

        # when gpu is on, .cpu() needed
        estimates = [estimate.detach().cpu().numpy() for estimate in estimates]

        for s in range(len(source_preds)):
            source_preds[s][:, :, start: start+output_length] = estimates[s]

            if get_betamap:
                for i, skip in enumerate(betas[s]):
                    beta_maps[s][i].append(skip.detach().cpu().numpy().squeeze())

    if get_betamap:
        return source_preds, beta_maps

    return source_preds


def estimate_track_musdb(model, track, input_length, get_betamap=False, save_folder=None):
    """
    separate one full track

    :param model:
    :param track: desired track to separate (tensor) : (bs=1, ch=1, t) or track object
    :param track: desired track to separate : track object
    :return: list of estimated sources : [ (1, 1, t), (1, 1, t) .. ]

    """
    # TODO : check needed - I used sf.read for training...

    # load track
    mix_audio, orig_sr, mix_channels = track.audio, track.rate, track.audio.shape[1]

    # mix_audio =
    mix_audio = Utils.ensure_sample_rate(mix_audio, desired_sample_rate=22050, file_sample_rate=orig_sr)
    if mix_audio.ndim > 1:
        mix_audio = (mix_audio[:, 0] + mix_audio[:, 1])/2.0
    track_length = mix_audio.shape[0]

    mix_audio = torch.from_numpy(mix_audio).unsqueeze(0).unsqueeze(0).float()

    source_preds = [np.zeros(mix_audio.shape, np.float32) for _ in range(2)]

    # save beta map
    if get_betamap:
        beta_maps = {s: [[] for _ in range(9)] for s in range(2)}

    for start in range(0, track_length, input_length):
        if start + input_length > track_length:
            start = track_length - input_length - 1

        mix_segment = mix_audio[:, :, start: start+input_length]

        # betas : tuple (beta1, beta2), beta1 : list of beta maps for skip connections, (bs, c, t)
        estimates, betas = model(mix_segment)

        # when gpu is on, .cpu() needed
        estimates = [estimate.detach().cpu().numpy() for estimate in estimates]

        for s in range(len(source_preds)):
            source_preds[s][:, :, start: start+input_length] = estimates[s]

            if get_betamap:
                for i, skip in enumerate(betas[s]):
                    beta_maps[s][i].append(skip.detach().cpu().numpy().squeeze())


    # Upsample predicted source audio and convert to stereo
    source_preds = [Utils.ensure_sample_rate(np.expand_dims(pred.squeeze(),1), orig_sr, 22050) for pred in source_preds]
    source_preds = [np.tile(pred, [1, mix_channels]) for pred in source_preds]

    estimates = {
        'vocals': source_preds[1],
        'accompaniment': source_preds[0]
    }

    if save_folder is not None:
        scores = museval.eval_mus_track(track, estimates, output_dir=save_folder)

        # print nicely formatted mean scores
        print(scores)

    if get_betamap:
        return source_preds, beta_maps

    return estimates


def compute_mean_metrics(json_folder, compute_averages=True):
    files = glob.glob(os.path.join(json_folder, "*.json"))
    sdr_inst_list = None
    for path in files:
        #print(path)
        with open(path, "r") as f:
            js = json.load(f)

        if sdr_inst_list is None:
            sdr_inst_list = [list() for _ in range(len(js["targets"]))]

        for i in range(len(js["targets"])):
            sdr_inst_list[i].extend([np.float(f['metrics']["SDR"]) for f in js["targets"][i]["frames"]])

    #return np.array(sdr_acc), np.array(sdr_voc)
    sdr_inst_list = [np.array(sdr) for sdr in sdr_inst_list]

    if compute_averages:
        return [(np.nanmedian(sdr), np.nanmedian(np.abs(sdr - np.nanmedian(sdr))), np.nanmean(sdr), np.nanstd(sdr)) for sdr in sdr_inst_list]
    else:
        return sdr_inst_list


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
    def __init__(self, config, target_wav):

        self.mode = config.mode
        self.data_path = config.data_path

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
        if self.mode == 'test':
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
            # source_pred, beta_maps = estimate_track(self.AttnNet, track, self.input_length, get_betamap=True)
            source_pred = estimate_track(self.AttnNet, track, self.input_length, get_betamap=False)

            for i, source in enumerate(source_pred):
                print(i)
                source = np.squeeze(source)

                save_path = os.path.join(save_folder, track_name + 'source{}.wav'.format(i))
                Utils.write_wav(source, save_path, 22050)

            # save beta maps
            # save_all_heatmaps(beta_maps, save_folder)


    def produce_musdb_source_estimates(self, subsets=None):
        '''
        Predicts source estimates for MUSDB for a given model checkpoint and configuration, and evaluate them.
        :param model_config: Model configuration of the model to be evaluated
        :param load_model: Model checkpoint path
        :return:
        '''

        save_folder = os.path.join(self.exp_path, self.saved_model.split('.')[0])

        mus = musdb.DB(root_dir=self.data_path)
        predict_fun = lambda track : estimate_track_musdb(self.AttnNet, track, self.input_length,
                                                          get_betamap=False, save_folder=save_folder)
        assert(mus.test(predict_fun))

        mus.run(predict_fun, estimates_dir=save_folder, subsets=subsets)


if __name__ == '__main__':
    path = '../log/attn_separation/attention10/132000-AttnNet/train'
    compute_mean_metrics(path)