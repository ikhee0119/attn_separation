import torch.nn as nn
import torch.nn.functional as f
import torch

from models.attention_modules import CrossAttention, SelfAttention


class OutputLayer(nn.Module):
    def __init__(self, attention_fn, dec_filter_size, emb_ch, skip_ch):
        super(OutputLayer, self).__init__()

        self.attention_fn = SelfAttention(emb_ch, skip_ch) if attention_fn == 'self_attention' \
            else CrossAttention(emb_ch, skip_ch)

        self.conv = nn.Conv1d(emb_ch+skip_ch, skip_ch, dec_filter_size, padding=dec_filter_size//2)
        self.centercrop = CenterCrop()
        self.activation = nn.Tanh()

    def forward(self, x, skip, is_sep=True):

        beta = None
        if is_sep:
            skip, beta = self.attention_fn(x, skip)

        skip = self.centercrop(x, skip)
        x = torch.cat((skip, x), 1) # channel-wise

        x = self.conv(x)
        # x = self.activation(x)

        return x, beta


class Interpolate(nn.Module):
    def __init__(self, context):
        super(Interpolate, self).__init__()

        self.context = context

    def forward(self, x):

        if self.context:
            length = x.size(-1)
            return f.interpolate(x, size=2*length-1, mode='linear', align_corners=True)

        else:
            return f.interpolate(x, scale_factor=2, mode='linear', align_corners=True)


class CenterCrop(nn.Module):
    def __init__(self):
        super(CenterCrop, self).__init__()

    def forward(self, x, skip):

        x_length = x.size(-1)
        skip_length = skip.size(-1)

        diff = (skip_length - x_length)//2
        assert(diff>=0)

        return skip[:, :, diff: x_length + diff]


class UpSkip(nn.Module):
    def __init__(self, attention_fn, dec_filter_size, emb_ch, skip_ch, is_skip_attention, context):
        super(UpSkip, self).__init__()

        self.attention_fn = SelfAttention(emb_ch, skip_ch) if attention_fn == 'self_attention' \
            else CrossAttention(emb_ch, skip_ch)

        self.is_skip_attention = is_skip_attention
        self.context = context

        self.upsample = Interpolate(self.context)
        self.centercrop = CenterCrop()

        self.conv = nn.Conv1d(emb_ch+skip_ch, skip_ch, dec_filter_size, padding=0) if context \
            else nn.Conv1d(emb_ch+skip_ch, skip_ch, dec_filter_size, padding=dec_filter_size//2)

        self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, skip):

        x = self.upsample(x)

        beta = None
        skip = self.centercrop(x, skip)

        if self.is_skip_attention:
            skip, beta = self.attention_fn(x, skip)

        x = torch.cat((skip, x), 1) # channel-wise

        x = self.conv(x)
        x = self.activation(x)

        return x, beta


class up(nn.Module):
    def __init__(self, dec_filter_size, in_ch, out_ch):
        super(up, self).__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.conv = nn.Conv1d(in_ch, out_ch, dec_filter_size, padding=dec_filter_size//2)
        self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):

        x = self.upsample(x)
        x = self.conv(x)
        x = self.activation(x)

        return x


class down(nn.Module):
    def __init__(self, enc_filter_size, in_ch, out_ch, context):
        super(down, self).__init__()

        self.conv = nn.Conv1d(in_ch, out_ch, enc_filter_size, padding=0) if context \
            else nn.Conv1d(in_ch, out_ch, enc_filter_size, padding=enc_filter_size//2)

        self.activation = nn.LeakyReLU(negative_slope=0.2)
        # self.pooling = nn.MaxPool1d(2)

    def forward(self, x):

        x = self.conv(x)
        x = self.activation(x)

        skip = x

        # x = self.pooling(x)
        # decimate : 1108
        x = x[:, :, ::2]

        return x, skip


class conv(nn.Module):
    def __init__(self, enc_filter_size, in_ch, out_ch, context):
        super(conv, self).__init__()

        self.conv = nn.Conv1d(in_ch, out_ch, enc_filter_size, padding=0) if context \
            else nn.Conv1d(in_ch, out_ch, enc_filter_size, padding=enc_filter_size//2)
        self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):

        x = self.conv(x)
        x = self.activation(x)

        return x