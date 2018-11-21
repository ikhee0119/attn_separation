import torch.nn as nn
import torch


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

        return emb1, emb2, beta, 1-beta


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

        return emb1, emb2, beta[:,:,:,0], beta[:,:,:,1]


class SelfAttention(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(SelfAttention, self).__init__()

        self.conv1 = nn.Conv1d(out_ch, out_ch, 15, padding=7)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 15, padding=7)
        self.conv3 = nn.Conv1d(out_ch, out_ch, 15, padding=7)
        self.conv4 = nn.Conv1d(out_ch, out_ch, 15, padding=7)

        self.activation = nn.LeakyReLU(negative_slope=0.2)
        self.map = nn.Sigmoid()

    def forward(self, x, skip=None, before_act=False):

        if skip is not None:
            x = skip

        x = self.conv1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.activation(x)

        x = self.conv3(x)
        x = self.activation(x)

        feature = self.conv4(x)

        if before_act:
            return feature

        beta = self.map(feature)
        x = torch.mul(beta, x)

        return x, beta