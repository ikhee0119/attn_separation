# reference : https://github.com/milesial/Pytorch-UNet

import torch
import torch.nn as nn


class AttnNet(nn.Module):
    def __init__(self, attention_fn, num_layers, num_filters, enc_filter_size, dec_filter_size):
        super(AttnNet, self).__init__()

        self.F = F(enc_filter_size, num_filters)

        self.conv1 = conv(enc_filter_size, num_filters*8, num_filters*9)

        self.attention_block = AttentionBlock(num_filters*9, num_filters*9)
        # self.attention_block = AttentionBlockSparse(num_filters*9, num_filters*9)

        self.G = G(attention_fn, dec_filter_size, num_filters)

        self.conv2 = conv(dec_filter_size, num_filters, 1)

    def forward(self, x, is_sep=True):

        if is_sep:

            # encoder

            x, skip_layers = self.F(x)
            x = self.conv1(x)

            # attention_module

            s1_emb, s2_emb, beta = self.attention_block(x)

            # decoder

            s1 = self.G(s1_emb, skip_layers, is_sep)
            s1 = self.conv2(s1)

            s2 = self.G(s2_emb, skip_layers, is_sep)
            s2 = self.conv2(s2)

            return s1, s2

        else:
            x, skip_layers = self.F(x)

            x = self.conv1(x)

            recon = self.G(x, skip_layers, is_sep)
            recon = self.conv2(recon)
            return recon


class F(nn.Module):
    def __init__(self, enc_filter_size, num_filters):
        super(F, self).__init__()

        self.down1 = down(enc_filter_size, 1, num_filters)
        self.down2 = down(enc_filter_size, num_filters, num_filters*2)
        self.down3 = down(enc_filter_size, num_filters*2, num_filters*3)
        self.down4 = down(enc_filter_size, num_filters*3, num_filters*4)
        self.down5 = down(enc_filter_size, num_filters*4, num_filters*5)
        self.down6 = down(enc_filter_size, num_filters*5, num_filters*6)
        self.down7 = down(enc_filter_size, num_filters*6, num_filters*7)
        self.down8 = down(enc_filter_size, num_filters*7, num_filters*8)

    def forward(self, x):

        x, skip1 = self.down1(x)
        x, skip2 = self.down2(x)
        x, skip3 = self.down3(x)
        x, skip4 = self.down4(x)
        x, skip5 = self.down5(x)
        x, skip6 = self.down6(x)
        x, skip7 = self.down7(x)
        x, skip8 = self.down8(x)

        return x, [skip1, skip2, skip3, skip4, skip5, skip6, skip7, skip8]


class G(nn.Module):
    def __init__(self, attention_fn, dec_filter_size, num_filters):
        super(G, self).__init__()

        self.up1 = UpSkip(attention_fn, dec_filter_size, num_filters*9, num_filters*8)
        self.up2 = UpSkip(attention_fn, dec_filter_size, num_filters*8, num_filters*7)
        self.up3 = UpSkip(attention_fn, dec_filter_size, num_filters*7, num_filters*6)
        self.up4 = UpSkip(attention_fn, dec_filter_size, num_filters*6, num_filters*5)
        self.up5 = UpSkip(attention_fn, dec_filter_size, num_filters*5, num_filters*4)
        self.up6 = UpSkip(attention_fn, dec_filter_size, num_filters*4, num_filters*3)
        self.up7 = UpSkip(attention_fn, dec_filter_size, num_filters*3, num_filters*2)
        self.up8 = UpSkip(attention_fn, dec_filter_size, num_filters*2, num_filters)

    def forward(self, x, skip_layers, is_sep=True):

        o = self.up1(x, skip_layers[-1], is_sep)
        o = self.up2(o, skip_layers[-2], is_sep)
        o = self.up3(o, skip_layers[-3], is_sep)
        o = self.up4(o, skip_layers[-4], is_sep)
        o = self.up5(o, skip_layers[-5], is_sep)
        o = self.up6(o, skip_layers[-6], is_sep)
        o = self.up7(o, skip_layers[-7], is_sep)
        o = self.up8(o, skip_layers[-8], is_sep)

        return o


class CrossAttention(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CrossAttention, self).__init__()

        self.f = nn.Conv1d(in_ch, out_ch, 1)
        self.g = nn.Conv1d(out_ch, out_ch, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x, skip=None):

        f = self.f(x)
        g = self.g(skip)

        beta = self.activation(torch.mul(f, g))
        o = torch.mul(beta, skip)

        return o, beta


class AttentionBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(AttentionBlock, self).__init__()

        self.attention_fn = SelfAttention(in_ch, out_ch)

    def forward(self, x, skip=None):

        emb1, beta = self.attention_fn(x)
        emb2 = x-emb1

        return emb1, emb2, beta


class AttentionBlockSparse(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(AttentionBlockSparse, self).__init__()

        self.attention_fn1 = SelfAttention(in_ch, out_ch)
        self.attention_fn2 = SelfAttention(in_ch, out_ch)

        # Todo : check
        self.softmax = nn.Softmax(dim=3)

    def forward(self, x, skip=None):

        f1 = self.attention_fn1(x, before_act=True)
        f2 = self.attention_fn2(x, before_act=True)

        # f1, f2 : (bs, c, t) => (bs, c, t, 1) for softmax
        beta = torch.cat((f1.unsqueeze(-1), f2.unsqueeze(-1)), -1)
        beta = self.softmax(beta)

        emb1 = torch.mul(x, beta[:,:,:,0])
        emb2 = torch.mul(x, beta[:,:,:,1])

        return emb1, emb2, beta


class SelfAttention(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(SelfAttention, self).__init__()

        self.conv = nn.Conv1d(out_ch, out_ch, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x, skip=None, before_act=False):

        if skip is not None:
            x = skip

        feature = self.conv(x)

        if before_act:
            return feature

        beta = self.activation(feature)
        x = torch.mul(beta, x)

        return x, beta


class UpSkip(nn.Module):
    def __init__(self, attention_fn, dec_filter_size, emb_ch, skip_ch):
        super(UpSkip, self).__init__()

        self.attention_fn = SelfAttention(emb_ch, skip_ch) if attention_fn == 'self_attention' \
            else CrossAttention(emb_ch, skip_ch)

        self.upsample = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.conv = nn.Conv1d(emb_ch+skip_ch, skip_ch, dec_filter_size, padding=dec_filter_size//2)
        self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, skip, is_sep=True):

        x = self.upsample(x)

        if is_sep:
            skip, beta = self.attention_fn(x, skip)

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
    attnnet = AttnNet(attention_fn='cross_attention', num_layers=8, num_filters=24, enc_filter_size=15,
                      dec_filter_size=5)

    x = torch.ones((1, 1, 16384))
    s1, s2 = attnnet(x)
    print(s1.size(), s2.size())