# reference : https://github.com/milesial/Pytorch-UNet

import torch
import torch.nn as nn
import torch.nn.functional as f


class AttnNet(nn.Module):
    def __init__(self, attention_fn, num_layers, num_filters, enc_filter_size, dec_filter_size,
                 is_skip_attention=True, is_diff=True, context=False, stereo=True):
        super(AttnNet, self).__init__()

        self.is_skip_attention = is_skip_attention
        self.is_diff = is_diff
        self.context = context

        self.F = F(enc_filter_size, num_filters, num_layers, context, stereo)
        self.conv1 = conv(enc_filter_size, num_filters*num_layers, num_filters*(num_layers+1), context)

        self.attention_block = AttentionBlock(num_filters*(num_layers+1), num_filters*(num_layers+1))
        # self.attention_block = AttentionBlockSparse(num_filters*9, num_filters*9)

        self.G = G(attention_fn, dec_filter_size, num_filters, num_layers, is_skip_attention, context, stereo)

        self.centercrop = CenterCrop()

    def forward(self, x):

        # if is_sep:

        # encoder
        mix = x
        x, skip_layers = self.F(x)
        x = self.conv1(x)

        # attention_module

        s1_emb, s2_emb, beta1, beta2 = self.attention_block(x)

        # decoder

        s1, betas1 = self.G(s1_emb, skip_layers)

        if self.is_diff:

            mix = self.centercrop(s1, mix)
            s2 = mix - s1
            # Todo : beta
            betas2 = betas1
        else:
            s2, betas2 = self.G(s2_emb, skip_layers)

        return (s1, s2), ([beta1] + betas1, [beta2] + betas2)


class F(nn.Module):
    def __init__(self, enc_filter_size, num_filters, num_layers, context, stereo):
        super(F, self).__init__()

        input_ch = 2 if stereo else 1

        self.downs = nn.ModuleList()
        self.downs.append(down(enc_filter_size, input_ch, num_filters, context=context))
        self.downs.extend([down(enc_filter_size, num_filters*layer, num_filters*(layer+1), context=context)
                           for layer in range(1, num_layers)])

    def forward(self, x):

        skips = []
        skips.append(x)
        for down in self.downs:
            x, skip = down(x)
            skips.append(skip)
        return x, skips


class G(nn.Module):
    def __init__(self, attention_fn, dec_filter_size, num_filters, num_layers,
                 is_skip_attention=True, context=True, stereo=True):
        super(G, self).__init__()

        self.is_skip_attention = is_skip_attention
        self.ups = nn.ModuleList(
            [UpSkip(attention_fn, dec_filter_size, num_filters*(layer+2), num_filters*(layer+1), is_skip_attention, context)
             for layer in reversed(range(num_layers))]
        )

        out_ch = 2 if stereo else 1
        self.output = OutputLayer(attention_fn, dec_filter_size, num_filters, out_ch)

    def forward(self, x, skip_layers):

        betas = []
        for i, up in enumerate(self.ups):
            x, beta = up(x, skip_layers[-i-1])
            betas.append(beta)

        x, beta = self.output(x, skip_layers[0], False)
        betas.append(beta)

        return x, betas


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


class OutputLayer(nn.Module):
    def __init__(self, attention_fn, dec_filter_size, emb_ch, skip_ch):
        super(OutputLayer, self).__init__()

        self.attention_fn = SelfAttention(emb_ch, skip_ch) if attention_fn == 'self_attention' \
            else CrossAttention(emb_ch, skip_ch)

        self.conv = nn.Conv1d(emb_ch+skip_ch, skip_ch, dec_filter_size, padding=dec_filter_size//2)
        self.centercrop = CenterCrop()

    def forward(self, x, skip, is_sep=True):

        beta = None
        if is_sep:
            skip, beta = self.attention_fn(x, skip)

        skip = self.centercrop(x, skip)
        x = torch.cat((skip, x), 1) # channel-wise

        x = self.conv(x)

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

if __name__ == '__main__':
    # (bs, c, t)
    attnnet = AttnNet(attention_fn='cross_attention', num_layers=12, num_filters=24, enc_filter_size=15,
                      dec_filter_size=5, context=True, stereo=True)

    # x = torch.ones((1, 1, 16384))
    # x = torch.ones((1, 1, 24563)) # num_layer=8
    # x = torch.ones((1, 1, 147443)) # num_layer=12
    x = torch.ones((1, 2, 147443)) # num_layer=12


    (s1, s2), (betas1, betas2) = attnnet(x)
    print(s1.size(), s2.size())