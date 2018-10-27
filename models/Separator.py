# reference : https://github.com/milesial/Pytorch-UNet
# Note : skip is connected after pooling is applied.

import torch
import torch.nn as nn

class AttnNet(nn.Module):
    def __init__(self, num_layers, num_filters, enc_filter_size, dec_filter_size):
        super(AttnNet, self).__init__()

        self.down1 = down(enc_filter_size, 1, num_filters)
        self.down2 = down(enc_filter_size, num_filters, num_filters*2)
        self.down3 = down(enc_filter_size, num_filters*2, num_filters*3)
        self.down4 = down(enc_filter_size, num_filters*3, num_filters*4)
        self.down5 = down(enc_filter_size, num_filters*4, num_filters*5)
        self.down6 = down(enc_filter_size, num_filters*5, num_filters*6)
        self.down7 = down(enc_filter_size, num_filters*6, num_filters*7)
        self.down8 = down(enc_filter_size, num_filters*7, num_filters*8)

        self.conv1 = conv(enc_filter_size, num_filters*8, num_filters*9)

        self.up1 = up(dec_filter_size, num_filters*9, num_filters*8)
        self.up2 = up(dec_filter_size, num_filters*8, num_filters*7)
        self.up3 = up(dec_filter_size, num_filters*7, num_filters*6)
        self.up4 = up(dec_filter_size, num_filters*6, num_filters*5)
        self.up5 = up(dec_filter_size, num_filters*5, num_filters*4)
        self.up6 = up(dec_filter_size, num_filters*4, num_filters*3)
        self.up7 = up(dec_filter_size, num_filters*3, num_filters*2)
        self.up8 = up(dec_filter_size, num_filters*2, num_filters)

        self.conv2 = conv(dec_filter_size, num_filters, 1)

    def forward(self, x):

        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x6 = self.down6(x5)
        x7 = self.down7(x6)
        emb = self.down8(x7)

        emb = self.conv1(emb)

        o = self.up1(emb)
        o = self.up2(o)
        o = self.up3(o)
        o = self.up4(o)
        o = self.up5(o)
        o = self.up6(o)
        o = self.up7(o)
        o = self.up8(o)

        o = self.conv2(o)

        return o

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
        x = self.pooling(x)

        return x

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
    attnnet = AttnNet(num_layers=8, num_filters=24, enc_filter_size=15,
                      dec_filter_size=5)

    x = torch.ones((1, 1, 16384))
    emb = attnnet(x)
    print(emb.size())