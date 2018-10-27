# reference : https://github.com/milesial/Pytorch-UNet

import torch
import torch.nn as nn


class AttnNet(nn.Module):
    def __init__(self, attention_fn, num_layers, num_filters, enc_filter_size, dec_filter_size):
        super(AttnNet, self).__init__()

        self.attention_fn = attention_fn

        self.down1 = down(enc_filter_size, 1, num_filters)
        self.down2 = down(enc_filter_size, num_filters, num_filters*2)
        self.down3 = down(enc_filter_size, num_filters*2, num_filters*3)
        self.down4 = down(enc_filter_size, num_filters*3, num_filters*4)
        self.down5 = down(enc_filter_size, num_filters*4, num_filters*5)
        self.down6 = down(enc_filter_size, num_filters*5, num_filters*6)
        self.down7 = down(enc_filter_size, num_filters*6, num_filters*7)
        self.down8 = down(enc_filter_size, num_filters*7, num_filters*8)

        self.conv1 = conv(enc_filter_size, num_filters*8, num_filters*9)
        self.attention = SelfAttention(num_filters*9, num_filters*9)

        self.up1 = UpSkip(attention_fn, dec_filter_size, num_filters*9, num_filters*8)
        self.up2 = UpSkip(attention_fn, dec_filter_size, num_filters*8, num_filters*7)
        self.up3 = UpSkip(attention_fn, dec_filter_size, num_filters*7, num_filters*6)
        self.up4 = UpSkip(attention_fn, dec_filter_size, num_filters*6, num_filters*5)
        self.up5 = UpSkip(attention_fn, dec_filter_size, num_filters*5, num_filters*4)
        self.up6 = UpSkip(attention_fn, dec_filter_size, num_filters*4, num_filters*3)
        self.up7 = UpSkip(attention_fn, dec_filter_size, num_filters*3, num_filters*2)
        self.up8 = UpSkip(attention_fn, dec_filter_size, num_filters*2, num_filters)

        self.conv2 = conv(dec_filter_size, num_filters, 1)

    def forward(self, x):

        x, skip1 = self.down1(x)
        x, skip2 = self.down2(x)
        x, skip3 = self.down3(x)
        x, skip4 = self.down4(x)
        x, skip5 = self.down5(x)
        x, skip6 = self.down6(x)
        x, skip7 = self.down7(x)
        x, skip8 = self.down8(x)

        x = self.conv1(x)
        x, beta = self.attention(x)

        o = self.up1(x, skip8)
        o = self.up2(o, skip7)
        o = self.up3(o, skip6)
        o = self.up4(o, skip5)
        o = self.up5(o, skip4)
        o = self.up6(o, skip3)
        o = self.up7(o, skip2)
        o = self.up8(o, skip1)

        o = self.conv2(o)

        return o


class SelfAttention(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(SelfAttention, self).__init__()

        self.conv = nn.Conv1d(in_ch, out_ch, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x, skip=None):

        if skip is not None:
            x = skip
        beta = self.activation(self.conv(x))
        x = torch.mul(beta, x)

        return x, beta


class UpSkip(nn.Module):
    def __init__(self, attention_fn, dec_filter_size, emb_ch, skip_ch):
        super(UpSkip, self).__init__()

        self.attention_fn = SelfAttention(skip_ch, skip_ch) if attention_fn == 'self_attention' else None

        self.upsample = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.conv = nn.Conv1d(emb_ch+skip_ch, skip_ch, dec_filter_size, padding=dec_filter_size//2)
        self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, skip):

        skip, beta = self.attention_fn(x, skip)
        x = self.upsample(x)

        x = torch.cat((skip, x), 1) # channel-wise

        x = self.conv(x)
        x = self.activation(x)

        return x


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
    def __init__(self, enc_filter_size, in_ch, out_ch):
        super(down, self).__init__()

        self.conv = nn.Conv1d(in_ch, out_ch, enc_filter_size, padding=enc_filter_size//2)
        self.activation = nn.LeakyReLU(negative_slope=0.2)

        self.pooling = nn.MaxPool1d(2)

    def forward(self, x):

        x = self.conv(x)
        x = self.activation(x)

        skip = x

        x = self.pooling(x)

        return x, skip


class conv(nn.Module):
    def __init__(self, enc_filter_size, in_ch, out_ch):
        super(conv, self).__init__()

        self.conv = nn.Conv1d(in_ch, out_ch, enc_filter_size, padding=enc_filter_size//2)
        self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):

        x = self.conv(x)
        x = self.activation(x)

        return x

if __name__ == '__main__':
    # (bs, c, t)

    from torch.autograd import Variable
    attnnet = AttnNet(attention_fn='self_attention', num_layers=8, num_filters=24, enc_filter_size=15,
                      dec_filter_size=5)

    x = torch.ones((1, 1, 16384))
    emb = attnnet(x)
    print(emb.size())