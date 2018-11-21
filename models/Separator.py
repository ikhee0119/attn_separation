# reference : https://github.com/milesial/Pytorch-UNet

import torch
import torch.nn as nn

from models.layers import OutputLayer, conv, CenterCrop, UpSkip, down
from models.attention_modules import AttentionBlock


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