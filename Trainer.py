from models.Separator import AttnNet
import torch
import torch.nn as nn
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

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.build_model()

    def build_model(self):

        self.AttnNet = AttnNet(attention_fn=self.attention_fn, num_layers=self.num_layers,
                               num_filters=self.num_filters, enc_filter_size=self.enc_filter_size,
                               dec_filter_size=self.dec_filter_size)

        self.optimizer = torch.optim.Adam(self.AttnNet.parameters(), self.lr, [0.5, 0.999])

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.optimizer.zero_grad()

    def train(self):

        lr = self.lr
        loss = nn.MSELoss()

        for epoch in range(self.epoch):

            # https://github.com/pytorch/pytorch/issues/5059
            # to make __getitem__ in data_loader random.
            np.random.seed()

            for (mix, accompany, vocal) in self.train_loader:

                mix = mix.float().to(self.device)
                accompany = accompany.float().to(self.device)
                vocal = vocal.float().to(self.device)

                estimated_accompany, estimated_vocal = self.AttnNet(mix)

                s1_loss = 0.5 * torch.mean((estimated_accompany-accompany)**2)
                s2_loss = 0.5 * torch.mean((estimated_vocal-vocal)**2)

                loss = s1_loss + s2_loss

                self.reset_grad()
                loss.backward()
                self.optimizer.step()