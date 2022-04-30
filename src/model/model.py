import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as tdist
import torchvision.models as models
import imp

import numpy as np

from .networks import Generator, MultiscaleDiscriminator, DenseD, Dis_Inn
# from .vae import VAE
from .loss import ColorLoss, PerceptualLoss, AdversarialLoss, KLDLoss
# StyleLoss, FeatureAvgLoss, MRFLoss
from ..utils import template_match, Adam16

import math


class BaseModel(nn.Module):
    def __init__(self, name, config, f):
        super(BaseModel, self).__init__()

        self.name = name
        self.config = config
        self.iteration = 0
        self.device = config.DEVICE
        self.g_weights_path = os.path.join(config.PATH, 'g.pth')
        self.d_weights_path = os.path.join(config.PATH, 'd.pth')
        self.dp_weights_path = os.path.join(config.PATH, 'dp.pth')

    def load(self):
        if self.name == 'PartPModel':
            if os.path.exists(self.g_weights_path):
                print('Loading %s Model ...' % self.name)

                g_data = torch.load(self.g_weights_path)
                self.g.load_state_dict(g_data['params'])
                self.iteration = g_data['iteration']

            if os.path.exists(self.d_weights_path):
                d_data = torch.load(self.d_weights_path)
                self.d.load_state_dict(d_data['params'])

            if os.path.exists(self.dp_weights_path):
                dp_data = torch.load(self.dp_weights_path)
                self.d_p.load_state_dict(dp_data['params'])
        
    
    def save(self, ite):
        print('\nSaving %s...\n' % self.name)
        if self.name == 'PartPModel':
            # print(self.name == 'PartPModel')
            torch.save({
                'iteration': self.iteration,
                'params': self.g.state_dict()}, self.g_weights_path + '_' + str(ite))
            torch.save({'params': self.d.state_dict()}, self.d_weights_path + '_' + str(ite))
            torch.save({'params': self.d_p.state_dict()}, self.dp_weights_path + '_' + str(ite))


class PartPModel(BaseModel):
    def __init__(self, config, f):
        super(PartPModel, self).__init__('PartPModel', config, f)

        g = Generator()
        d_p = MultiscaleDiscriminator()
        d = DenseD()

        color_loss = ColorLoss()
        adversarial_loss = AdversarialLoss()
        l1_loss = nn.L1Loss()
        kld_loss = KLDLoss()

        content_loss = PerceptualLoss()
        
        self.add_module('g', g)
        self.add_module('d', d)
        self.add_module('d_p', d_p)

        self.add_module('content_loss', content_loss)
        self.add_module('adversarial_loss', adversarial_loss)
        self.add_module('color_loss', color_loss)
        self.add_module('l1_loss', l1_loss)
        self.add_module('kld_loss', kld_loss)
        
        self.g_optimizer = Adam16(params=g.parameters(), lr=float(config.G_LR), betas=(config.BETA1, config.BETA2), weight_decay=0.0, eps=1e-8)
        self.d_optimizer = Adam16(params=d.parameters(), lr=float(config.D_LR), betas=(config.BETA1, config.BETA2), weight_decay=0.0, eps=1e-8)
        self.d_p_optimizer = Adam16(params=d_p.parameters(), lr=float(config.D_LR), betas=(config.BETA1, config.BETA2), weight_decay=0.0, eps=1e-8)
        # self.g_optimizer = optim.Adam(params=g.parameters(), lr=float(config.LR / 2), betas=(config.BETA1, config.BETA2))
        # self.d_optimizer = optim.Adam(params=d.parameters(), lr=float(config.LR * 2), betas=(config.BETA1, config.BETA2))
        # self.d_p_optimizer = optim.Adam(params=d_p.parameters(), lr=float(config.LR * 2), betas=(config.BETA1, config.BETA2))
        
    def process(self, data, pdata, half_fmask, ite):
        self.iteration += 1

        self.ite = ite

        mask = 1 - half_fmask
        # zero optimizers
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()
        self.d_p_optimizer.zero_grad()

        o, (mu, logvar), ys = self.g(pdata, mask)

        kld_loss = self.kld_loss(mu, logvar) * self.config.KLD_LOSS_WEIGHT

        # coarse loss
        c_loss = 0
        f_loss = 0

        # total
        g_loss = 0
        d_p_loss = 0
        d_loss = 0
        # ---------------------------- G / D ----------------------------
        # d loss
        d_real = data
        d_fake = o.detach()

        # d_real_arr = self.d(torch.cat((d_real, pdata), dim=1))
        # d_fake_arr = self.d(torch.cat((d_fake, pdata), dim=1))
        # g_fake_arr = self.d(torch.cat((o, pdata), dim=1))

        # g_adv = 0
        # for i in range(len(d_real_arr)):
        #     d_real_l = self.adversarial_loss(d_real_arr[i], True, True)
        #     d_fake_l = self.adversarial_loss(d_fake_arr[i], False, True)
        #     d_loss += (d_real_l + d_fake_l) / 2

        #     g_adv += self.adversarial_loss(g_fake_arr[i], True, False) * self.config.G1_ADV_LOSS_WEIGHT

        # d_loss = d_loss / 2
        # f_loss += g_adv / 2

        g_adv = 0

        # cat 3 + 3
        # d_real_6 = self.d(torch.cat((d_real, pdata), dim=1))
        # d_fake_6 = self.d(torch.cat((d_fake, pdata), dim=1))
        # g_fake = self.d(torch.cat((o, pdata), dim=1))
        d_real_6 = self.d(d_real)
        d_fake_6 = self.d(d_fake)
        g_fake = self.d(o)

        d_real_l = self.adversarial_loss(d_real_6, True, True)
        d_fake_l = self.adversarial_loss(d_fake_6, False, True)
        d_loss += (d_real_l + d_fake_l) / 2

        g_adv += self.adversarial_loss(g_fake, True, False) * self.config.G1_ADV_LOSS_WEIGHT

        # cat 3
        d_real_arr_p = self.d_p(d_real * half_fmask)
        d_fake_arr_p = self.d_p(d_fake * half_fmask)
        g_fake_arr_p = self.d_p(o * half_fmask)

        # d_real_arr_p = self.d_p(d_real)
        # d_fake_arr_p = self.d_p(d_fake)
        # g_fake_arr_p = self.d_p(o)

        g_p_adv = 0
        for i in range(len(d_real_arr_p)):
            d_p_real_l = self.adversarial_loss(d_real_arr_p[i], True, True)
            d_p_fake_l = self.adversarial_loss(d_fake_arr_p[i], False, True)
            d_p_loss += (d_p_real_l + d_p_fake_l) / 2

            g_p_adv += self.adversarial_loss(g_fake_arr_p[i], True, False)

        d_p_loss = d_p_loss / 2
        g_adv += g_p_adv / 2 * self.config.G1_ADV_LOSS_WEIGHT

        # g_adv += (self.adversarial_loss(g_fake, True, False) + self.adversarial_loss(g_fake_p, True, False))  * self.config.G1_ADV_LOSS_WEIGHT

        f_loss += g_adv / 2

        # g_1 content loss
        g_content_loss, g_sty_loss = self.content_loss(o, data)
        g_content_loss = g_content_loss * self.config.G1_CONTENT_LOSS_WEIGHT
        g_sty_loss = g_sty_loss * self.config.G2_STYLE_LOSS_WEIGHT
        c_loss += g_content_loss
        f_loss += g_sty_loss

        g_color_loss = self.color_loss(o, data)
        g_color_loss = g_color_loss * self.config.G1_COLOR_LOSS_WEIGHT
        c_loss += g_color_loss

        # g l1 loss
        g_l1_loss = self.l1_loss(o, data) * self.config.G2_L1_LOSS_WEIGHT
        c_loss += g_l1_loss

        g_loss = kld_loss + c_loss + f_loss
        # g_loss = c_loss + f_loss
        
        logs = [
            ("l_d", d_loss.item()),
            ("l_dp", d_p_loss.item()),
            ("l_g_adv", g_adv.item()),
            ("l_g_con", g_content_loss.item()),
            ("l_color", g_color_loss.item()),
            ("l_l1", g_l1_loss.item()),
            ("l_kld", kld_loss.item()),
            ("l_sty", g_sty_loss.item())
        ]
        # return o, d_loss, d_p_loss, c_loss + g_sty_loss, logs

        return o, d_loss, d_p_loss, g_loss, logs
        # return o, None, d_p_loss, g_loss, logs
    
    def forward(self, pdata, half_fmask, pos=None, z=None):
        o, (mu, logvar), ys = self.g(pdata, 1 - half_fmask)
        return o, ys

    def backward(self, d_loss, d_p_loss, g_loss):
        # if self.ite > self.config.COARSE_ITE:
        d_loss.backward()
        self.d_optimizer.step()

        d_p_loss.backward()
        self.d_p_optimizer.step()

        g_loss.backward()
        self.g_optimizer.step()
