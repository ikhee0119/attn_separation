from models.Separator import AttnNet
import torch
import numpy as np
import time
import datetime
import os


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

        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay

        self.log_step = config.log_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.exp_path = config.exp_path

        self.build_model()
        self.use_tensorboard = config.use_tensorboard

        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):

        self.AttnNet = AttnNet(attention_fn=self.attention_fn, num_layers=self.num_layers,
                               num_filters=self.num_filters, enc_filter_size=self.enc_filter_size,
                               dec_filter_size=self.dec_filter_size)

        self.optimizer = torch.optim.Adam(self.AttnNet.parameters(), self.lr, [0.5, 0.999])

        self.AttnNet.to(self.device)

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.exp_path)

    def update_lr(self, lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.optimizer.zero_grad()

    def train(self):

        lr = self.lr
        step = 0

        start_time = time.time()

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

                logs = {
                    'loss/total': loss.item(),
                    'loss/acc': s1_loss.item(),
                    'loss/voc': s2_loss.item()}

                if (step+1) % self.log_step == 0:

                    et = time.time() - start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    log = "Elapsed [{}], Iteration [{}/{}]".format(et, step + 1, self.num_iters)
                    for tag, value in logs.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)

                    if self.use_tensorboard:
                        for tag, value in logs.items():
                            self.logger.scalar_summary(tag, value, step+1)

                # Save model checkpoints.
                if (step + 1) % self.model_save_step == 0:
                    save_path = os.path.join(self.exp_path, '{}-AttnNet.ckpt'.format(step + 1))
                    torch.save(self.AttnNet.state_dict(), save_path)
                    print('Saved model checkpoints into {}...'.format(save_path))

                # Decay learning rates.
                if (step + 1) % self.lr_update_step == 0 and (step + 1) > (self.num_iters - self.num_iters_decay):
                    lr -= (self.lr / float(self.num_iters_decay))
                    self.update_lr(lr)
                    print('Decayed learning rates, lr: {}.'.format(lr))

                step += 1


    def test(self):
        pass