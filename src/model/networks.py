import torch
import torch.nn as nn
import functools
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
from .partialconv import PartialConv2d
from .patchtransmit import PatchTransmit
import torch.nn.utils.spectral_norm as spectral_norm

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='xavier', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)


class Generator(BaseNetwork):

    def __init__(self):
        super().__init__()

        self.ngf = 64
        nf = self.ngf
        self.sw, self.sh = 8, 8

        self.z_dim = 256
        self.fc = nn.Linear(self.z_dim, 16 * nf * self.sw * self.sh)

        self.Encoder = Encoder()

        # without noise
        # self.head_0 = ResnetBlock(8 * nf, 16 * nf, d = 2)
        # with noise
        self.head_0 = ResnetBlock(16 * nf, 16 * nf, d = 2)

        self.G_middle_0 = ResnetBlock(16 * nf, 16 * nf, d = 2)
        self.G_middle_1 = ResnetBlock(16 * nf, 16 * nf, d = 2)

        self.up_0 = ResnetBlock(16 * nf, 8 * nf, d = 4)
        self.up_1 = ResnetBlock(8 * nf, 4 * nf, d = 4)
        self.up_2 = ResnetBlock(4 * nf, 2 * nf, d = 4)
        self.up_3 = ResnetBlock(2 * nf, 1 * nf)

        final_nc = nf

        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)

        self.tanh = nn.Tanh()
        # 28 for train & test
        # self.patchtransmit = PatchTransmit(nf*2)
        # 29 for train & test
        self.patchtransmit = PatchTransmit(nf)

        self.init_weights()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        # print(mu, logvar)
        
        return eps.mul(std) + mu

    def adain(self, f, z, eps=1e-6):
        b, c = f.size()[:2]

        f_view = f.view(b, c, -1)
        z_view = z.view(b, c, -1)

        f_mean = f_view.mean(dim=2).view(b, c, 1, 1)
        f_std = f_view.std(dim=2).view(b, c, 1, 1) + eps

        z_mean = z_view.mean(dim=2).view(b, c, 1, 1)
        z_std = z_view.std(dim=2).view(b, c, 1, 1) + eps

        z = f_std * (z - z_mean) / z_std + f_mean

        return z

    def forward(self, input, mask, z=None):
        mask_image = input

        [x_0, x_1, x_2, x_3, x_4, z_in], [mask_0, mask_1, mask_2, mask_3, mask_4], (mu, logvar) = self.Encoder(mask_image, mask)

        # part-noise
        z = self.reparameterize(mu, logvar)
        # normal noise
        # z = torch.randn(input.size(0), self.z_dim, dtype=torch.float32).cuda()
        # with/without noise
        z = self.fc(z)
        x = z.view(-1, 16 * self.ngf, self.sh, self.sw)
        y_1 = self.head_0(x, x_4, mask_4)
        # without noise
        # y_1 = self.head_0(x_4)
        # y_1 = self.head_0(x)
        # Local Adain
        # x = self.local_adain(x, mask_4)
        x = self.up(y_1)
        # 16, 16
        y_2_1 = self.G_middle_0(x, x_3, mask_3)
        # y_2_1 = self.G_middle_0(x)
        # x = self.local_adain(x, mask_3)

        y_2_2 = self.G_middle_1(y_2_1, x_3, mask_3)
        # y_2_2 = self.G_middle_1(y_2_1)
        # x = self.local_adain(x, mask_3)

        x = self.up(y_2_2)
        # 16, 32
        y_3 = self.up_0(x, x_2, mask_2)
        # y_3 = self.up_0(x)
        # x = self.local_adain(x, mask_2)
        x = self.up(y_3)
        # 8, 64
        y_4 = self.up_1(x, x_1, mask_1)
        # y_4 = self.up_1(x)
        # x = self.local_adain(x, mask_1)
        # Local Adain
        # f_a_2 = self.local_adain(f_a_2, masks[0], masks[2])

        x = self.up(y_4)
        # 4, 128
        y_5 = self.up_2(x, x_0, mask_0, isprint=True)
        # y_5_a, _ = self.patchtransmit(y_5, mask_0)

        x = self.up(y_5)
        # 2, 256
        y_6 = self.up_3(x)
        # y_6 = self.up_3(x)
        y_6_a, _, w1, w2 = self.patchtransmit(y_6, mask)
        # y_6_a = self.patchtransmit(y_6, mask)

        # x = self.conv_img(F.leaky_relu(y_6_a, 2e-1))
        x = self.conv_img(y_6_a)
        x = (self.tanh(x) + 1) / 2
        # x = self.tanh(x)

        # return x, (mu, logvar), (y_1, y_2_1, y_2_2, y_3, y_4, y_5, _, y_5_a, y_6)
        # y1 = torch.mean(y_6, dim=1, keepdim=True)
        # y2 = torch.mean(y_6_a, dim=1, keepdim=True)
        # y3 = torch.mean(_, dim=1, keepdim=True)
        # y4 = torch.mean(w1, dim=1, keepdim=True)
        # y5 = torch.mean(w2, dim=1, keepdim=True)
        # min = y3.min()
        # max = y3.max()
        # return x, (mu, logvar), (y1 - min / max, (y2 - min) / max, (y3 - min) / max, (y4 - min) / max, (y5 - min) / max )
        return x, (mu, logvar), None

class ResnetBlock(BaseNetwork):
    def __init__(self, fin, fout, kernel_size=3, d=2):
        super().__init__()

        fmiddle = min(fin, fout)
        self.d = d

        pw = (kernel_size - 1) // 2
        self.relu = nn.LeakyReLU(0.2, False)
        # self.relu = nn.ReLU()

        self.conv_block_1 = nn.Sequential(
            nn.ReflectionPad2d(pw),
            nn.Conv2d(fin, fmiddle, kernel_size=kernel_size),
        )

        self.conv_block_2 = nn.Sequential(
            nn.ReflectionPad2d(pw),
            nn.Conv2d(fmiddle, fout, kernel_size=kernel_size)
        )

        self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)
        self.norm_1 = FeatureTransfer(fmiddle)
        self.norm_2 = FeatureRec(fmiddle)

        self.conv_1_1 = nn.Conv2d(fin+fin//2, fin, kernel_size=1, bias=False)
        self.conv_1_2 = nn.Conv2d(fin+fin//4, fin, kernel_size=1, bias=False)

        self.init_weights()
        self.inorm = nn.InstanceNorm2d(fout)

        self.tanh = nn.Tanh()

    def forward(self, x, t=None, mask=None, isprint=False):
        x_s = self.conv_s(x)
        # x_s = self.inorm(x_s)
        x_s = self.relu(x_s)

        d = self.d

        # 18, 19
        if t is not None:
            # print(x.shape, t.shape)
            x = torch.cat([x, t], 1)
            if d == 2:
                x = self.conv_1_1(x)
                x = self.relu(x)
            if d == 4:
                x = self.conv_1_2(x)
                x = self.relu(x)

        y = self.conv_block_1(x)
        if t is not None:
            y, g = self.norm_1(y, mask)
            # 32
            # y = yy + y
            y, gamma, beta = self.norm_2(y, mask, g)
        # y = self.inorm(y)
        y = self.relu(y)

        y = self.conv_block_2(y)
        # if t is not None:
        #     y = self.norm_2(y, mask, f_h)
        y = self.relu(y)
        # print(gamma, beta)

        out = x_s + y
        return out

def calc_mean_std(f, eps=1e-6):
    b, c = f.size()[:2]
    f_mean = f.mean(dim=2).view(b, c, 1, 1)
    f_std = f.std(dim=2).view(b, c, 1, 1) + eps

    return f_mean, f_std

class FeatureTransfer(nn.Module):
    def __init__(self, out_nc):
        super(FeatureTransfer, self).__init__()
        self.norm = nn.InstanceNorm2d(out_nc, affine=False)

    def forward(self, f, mask):
        [b, c, h, w] = f.size()

        # f = self.norm(f)
        # _, f_std = calc_mean_std(f)

        f = f.view(b, c, -1)
        
        mask_one_channel = mask.view(b, 1, -1)[0][0]
        index_good = torch.nonzero(mask_one_channel)
        index_bad = torch.nonzero(1 - mask_one_channel)

        f_local = f[:, :, index_good]
        f_global = f[:, :, index_bad]

        # print(f_global.shape)
        # print(torch.mean(f_global[0][60]), torch.var(f_global[0][60]))
        # print(torch.mean(f_local[0][60]), torch.var(f_local[0][60]))

        # f_global = self.norm(f_global)
        f_global_mean, f_global_std = calc_mean_std(f_global)
        f_global = (f_global - f_global_mean) / f_global_std

        f_local_mean, f_local_std = calc_mean_std(f_local)

        # x = self.norm(f_global)
        f_global_adain = f_global * f_local_std + f_local_mean

        f[:, :, index_bad] = f_global_adain
        # print(torch.mean(f_local_mean), torch.mean(f_local_std))

        f = f.view(b, c, h, w)

        return f, f_global

class FeatureRec(BaseNetwork):
    def __init__(self, in_nc):
        super(FeatureRec, self).__init__()

        ks = 1
        pw = 0
        nhidden = 128

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(in_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, in_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, in_nc, kernel_size=ks, padding=pw)
        self.init_weights()

    def forward(self, f, mask, f_1):
        [b, c, h, w] = f.size()

        f = f.view(b, c, -1)
        
        mask_one_channel = mask.view(b, 1, -1)[0][0]
        index_bad = torch.nonzero(1 - mask_one_channel)

        f_global = f[:, :, index_bad]
        # print(torch.mean(f_global[0][60]), torch.var(f_global[0][60]))
        # print(torch.mean(f_1), torch.var(f_1))
        f_mlp = self.mlp_shared(f_1)
        gamma = self.mlp_gamma(f_mlp)
        beta = self.mlp_beta(f_mlp)

        f_global = f_global * (1 + gamma) + beta

        # print(torch.mean(f_global[0][60]), torch.var(f_global[0][60]))
        f[:, :, index_bad] = f_global

        f = f.view(b, c, h, w)

        return f, torch.mean(gamma), torch.mean(beta)

class Encoder(BaseNetwork):
    def __init__(self, in_channels=3, nf=64, use_spectral_norm=True, init_weights=True):
        super(Encoder, self).__init__()

        self.nf = nf

        # 1, 128
        self.pc_0 = PartialConv2d(in_channels=in_channels, out_channels=nf, kernel_size=3, stride=2, padding=1)
        # 2, 64
        self.pc_1 = PartialConv2d(in_channels=nf, out_channels=nf*2, kernel_size=3, stride=2, padding=1)
        # 4, 32
        self.pc_2 = PartialConv2d(in_channels=nf*2, out_channels=nf*4, kernel_size=3, stride=2, padding=1)
        # 8, 16
        self.pc_3 = PartialConv2d(in_channels=nf*4, out_channels=nf*8, kernel_size=3, stride=2, padding=1)
        # 8, 8
        self.pc_4 = PartialConv2d(in_channels=nf*8, out_channels=nf*8, kernel_size=3, stride=2, padding=1)
        # 8, 4
        self.pc_5 = PartialConv2d(in_channels=nf*8, out_channels=nf*8, kernel_size=3, stride=2, padding=1)

        # Linear
        self.fc_mu = nn.Linear(nf * 8 * 4 * 4, 256)
        self.fc_var = nn.Linear(nf * 8 * 4 * 4, 256)

        self.relu = nn.LeakyReLU(0.2, False)
        self.down = nn.UpsamplingNearest2d(scale_factor=.5)

        if init_weights:
            self.init_weights()

    def norm(self, f, mask, nc):
        # [b, c, h, w] = f.size()

        # f = f.view(b, c, -1)
        
        # mask_one_channel = mask.view(b, 1, -1)[0][0]
        # index_good = torch.nonzero(mask_one_channel)

        # f_local = f[:, :, index_good]

        # norm = nn.InstanceNorm2d(nc, affine=False)
        # f_local = norm(f_local)

        # f[:, :, index_good] = f_local

        # f = f.view(b, c, h, w)
        norm = nn.InstanceNorm2d(nc)
        f = norm(f)
        return f

     
    def forward(self, x, mask):
        nf = self.nf

        x_0, mask_0 = self.pc_0(x, mask)
        x_0 = self.norm(x_0, mask_0, nf)
        x_0 = self.relu(x_0)

        mask_c_0 = self.down(mask)
        x_0 = x_0 * mask_c_0

        x_1, mask_1 = self.pc_1(x_0, mask_0)
        x_1 = self.norm(x_1, mask_1, nf*2)
        x_1 = self.relu(x_1)

        mask_c_1 = self.down(mask_c_0)
        x_1 = x_1 * mask_c_1

        x_2, mask_2 = self.pc_2(x_1, mask_1)
        x_2 = self.norm(x_2, mask_2, nf*2)
        x_2 = self.relu(x_2)

        mask_c_2 = self.down(mask_c_1)
        x_2 = x_2 * mask_c_2

        x_3, mask_3 = self.pc_3(x_2, mask_2)
        x_3 = self.norm(x_3, mask_3, nf*4)
        x_3 = self.relu(x_3)

        mask_c_3 = self.down(mask_c_2)
        x_3 = x_3 * mask_c_3

        x_4, mask_4 = self.pc_4(x_3, mask_3)
        x_4 = self.norm(x_4, mask_4, nf*4)
        x_4 = self.relu(x_4)

        mask_c_4 = self.down(mask_c_3)
        x_4 = x_4 * mask_c_4

        z, mask_5 = self.pc_5(x_4, mask_4)
        z = self.norm(z, mask_5, nf*4)
        z = self.relu(z)

        o = z.view(z.size(0), -1)

        mu = self.fc_mu(o)
        logvar = self.fc_var(o)

        return [x_0, x_1, x_2, x_3, x_4, z], [mask_c_0, mask_c_1, mask_c_2, mask_c_3, mask_c_4], (mu, logvar)

class MultiscaleDiscriminator(BaseNetwork):

    def __init__(self):
        super().__init__()
        
        self.num_d = 2

        self.d_1 = NLayerDiscriminator()
        self.d_2 = NLayerDiscriminator()

        self.init_weights()

    def downsample(self, input):
        return F.avg_pool2d(input, kernel_size=3,
                            stride=2, padding=[1, 1],
                            count_include_pad=False)

    def forward(self, input):
        get_intermediate_features = False

        out_1 = self.d_1(input)
        input = self.downsample(input)
        out_2 = self.d_2(input)

        return [out_1, out_2]

class NLayerDiscriminator(BaseNetwork):

    def __init__(self, in_channels=3):
        super().__init__()
        self.n_layers_D = 4

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        nf = 64
        input_nc = in_channels

        sequence = [[nn.Conv2d(input_nc, nf, kernel_size=kw, stride=2, padding=padw),
                     nn.LeakyReLU(0.2, False)]]

        for n in range(1, self.n_layers_D):
            nf_prev = nf
            nf = min(nf * 2, 512)
            stride = 1 if n ==  self.n_layers_D - 1 else 2
            sequence += [[spectral_norm(nn.Conv2d(nf_prev, nf, kernel_size=kw,
                                               stride=stride, padding=padw)),
                          nn.LeakyReLU(0.2, False)
                          ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        # We divide the layers into groups to extract intermediate layer outputs
        for n in range(len(sequence)):
            self.add_module('model' + str(n), nn.Sequential(*sequence[n]))

    def forward(self, input):
        results = [input]
        for submodel in self.children():
            intermediate_output = submodel(results[-1])
            results.append(intermediate_output)

        get_intermediate_features = False
        if get_intermediate_features:
            return results[1:]
        else:
            return torch.sigmoid(results[-1])
        
class Dis_Inn(BaseNetwork):
    def __init__(self, in_channels=3, use_sigmoid=True, init_weights=True):
        super(Dis_Inn, self).__init__()
        self.use_sigmoid = use_sigmoid
        
        self.conv1 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1)),
        )

        if init_weights:
            self.init_weights()
     
    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size*growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

class DenseD(nn.Module):
    def __init__(self, growth_rate=32, block_config=(3, 3, 3),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):

        super(DenseD, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # output layer
        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=1, kernel_size=4, stride=1, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        out = torch.sigmoid(out)
        return out


class PatchDiscriminator(BaseNetwork):
    def __init__(self):
        super(PatchDiscriminator, self).__init__()
        # Down sampling
        in_channels = 3
        latent_channels = 64
        pad_type = 'zero'
        activation = 'lrelu'
        norm = 'none'

        self.block1 = Conv2dLayer(in_channels, latent_channels, 7, 1, 3, pad_type = pad_type, activation = activation, norm = norm, sn = True)
        self.block2 = Conv2dLayer(latent_channels, latent_channels * 2, 4, 2, 1, pad_type = pad_type, activation = activation, norm = norm, sn = True)
        self.block3 = Conv2dLayer(latent_channels * 2, latent_channels * 4, 4, 2, 1, pad_type = pad_type, activation = activation, norm = norm, sn = True)
        self.block4 = Conv2dLayer(latent_channels * 4, latent_channels * 4, 4, 2, 1, pad_type = pad_type, activation = activation, norm = norm, sn = True)
        self.block5 = Conv2dLayer(latent_channels * 4, latent_channels * 4, 4, 2, 1, pad_type = pad_type, activation = activation, norm = norm, sn = True)
        self.block6 = Conv2dLayer(latent_channels * 4, 1, 4, 2, 1, pad_type = pad_type, activation = 'none', norm = 'none', sn = True)

        self.init_weights()
        
    def forward(self, img):
        # the input x should contain 4 channels because it is a combination of recon image and mask
        x = img
        x = self.block1(x)                                      # out: [B, 64, 256, 256]
        x = self.block2(x)                                      # out: [B, 128, 128, 128]
        x = self.block3(x)                                      # out: [B, 256, 64, 64]
        x = self.block4(x)                                      # out: [B, 256, 32, 32]
        x = self.block5(x)                                      # out: [B, 256, 16, 16]
        x = self.block6(x)                                      # out: [B, 256, 8, 8]
        x = torch.sigmoid(x)
        return x


class Conv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, pad_type = 'zero', activation = 'elu', norm = 'none', sn = False):
        super(Conv2dLayer, self).__init__()
        # Initialize the padding scheme
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)
        
        # Initialize the normalization type
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == 'ln':
            self.norm = LayerNorm(out_channels)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)
        
        # Initialize the activation funtion
        if activation == 'relu':
            self.activation = nn.ReLU(inplace = True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace = True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif activation == 'selu':
            self.activation = nn.SELU(inplace = True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # Initialize the convolution layers
        if sn:
            self.conv2d = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation))
        else:
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation)
    
    def forward(self, x):
        x = self.pad(x)
        x = self.conv2d(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x