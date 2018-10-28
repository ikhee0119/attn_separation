from models.Separator import AttnNet
import torch
import numpy as np

class Trainer:
    def __init__(self, train_loader, config):

        self.train_loader = train_loader

        self.attention_fn = config.attention_fn

        self.num_sources = config.num_sources

        self.num_layers = config.num_layers
        self.num_filters = config.num_filters
        self.input_length = config.input_length

        self.enc_filter_size = config.enc_filter_size
        self.dec_filter_size = config.dec_filter_size

        self.epoch = config.epoch
        self.lr = config.lr

        self.build_model()

    def build_model(self):

        self.AttnNet = AttnNet(attention_fn=self.attention_fn, num_layers=self.num_layers,
                               num_filters=self.num_filters, enc_filter_size=self.enc_filter_size,
                               dec_filter_size=self.dec_filter_size)

        self.optimizer = torch.optim.Adam(self.AttnNet.parameters(), self.lr, [0.5, 0.999])

    def train(self):

        for epoch in range(self.epoch):
            # https://github.com/pytorch/pytorch/issues/5059
            np.random.seed()
            for (mix, accompany, vocal) in self.train_loader:
                print(mix[0])