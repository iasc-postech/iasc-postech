# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# models/deep_big_resnet.py

from this import d
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.ops as ops

class MODULES(object):
    def __init__(self, apply_g_sn=True, apply_d_sn=True, g_act_fn='ReLU', d_act_fn='ReLU'):
        self.apply_g_sn = apply_g_sn
        self.apply_d_sn = apply_d_sn
        self.g_act_fn = g_act_fn
        self.d_act_fn = d_act_fn
        if self.apply_g_sn:
            self.g_conv2d = ops.snconv2d
            self.g_deconv2d = ops.sndeconv2d
            self.g_linear = ops.snlinear
            self.g_embedding = ops.sn_embedding
        else:
            self.g_conv2d = ops.conv2d
            self.g_deconv2d = ops.deconv2d
            self.g_linear = ops.linear
            self.g_embedding = ops.embedding

        if self.apply_d_sn:
            self.d_conv2d = ops.snconv2d
            self.d_deconv2d = ops.sndeconv2d
            self.d_linear = ops.snlinear
            self.d_embedding = ops.sn_embedding
        else:
            self.d_conv2d = ops.conv2d
            self.d_deconv2d = ops.deconv2d
            self.d_linear = ops.linear
            self.d_embedding = ops.embedding

        self.g_bn = ops.ConditionalBatchNorm2d

        if not self.apply_d_sn:
            self.d_bn = ops.batchnorm_2d


        if self.g_act_fn == "ReLU":
            self.g_act_fn = nn.ReLU(inplace=True)
        elif self.g_act_fn == "Leaky_ReLU":
            self.g_act_fn = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif self.g_act_fn == "ELU":
            self.g_act_fn = nn.ELU(alpha=1.0, inplace=True)
        elif self.g_act_fn == "GELU":
            self.g_act_fn = nn.GELU()
        elif self.g_act_fn == "Auto":
            pass
        else:
            raise NotImplementedError

        if self.d_act_fn == "ReLU":
            self.d_act_fn = nn.ReLU(inplace=True)
        elif self.d_act_fn == "Leaky_ReLU":
            self.d_act_fn = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif self.d_act_fn == "ELU":
            self.d_act_fn = nn.ELU(alpha=1.0, inplace=True)
        elif self.d_act_fn == "GELU":
            self.d_act_fn = nn.GELU()
        elif self.g_act_fn == "Auto":
            pass
        else:
            raise NotImplementedError

class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, affine_input_dim, MODULES):
        super(GenBlock, self).__init__()

        self.bn1 = MODULES.g_bn(affine_input_dim, in_channels, MODULES)
        self.bn2 = MODULES.g_bn(affine_input_dim, out_channels, MODULES)

        self.activation = MODULES.g_act_fn
        self.conv2d0 = MODULES.g_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        self.conv2d1 = MODULES.g_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2d2 = MODULES.g_conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, affine):
        x0 = x
        x = self.bn1(x, affine)
        x = self.activation(x)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv2d1(x)

        x = self.bn2(x, affine)
        x = self.activation(x)
        x = self.conv2d2(x)

        x0 = F.interpolate(x0, scale_factor=2, mode="nearest")
        x0 = self.conv2d0(x0)
        out = x + x0
        return out


class Generator(nn.Module):
    def __init__(self, z_dim, g_shared_dim, img_size, g_conv_dim, apply_attn, attn_g_loc, num_classes, MODULES,
     extra_feature_condition=False):
        super(Generator, self).__init__()
        g_in_dims_collection = {
            "32": [g_conv_dim * 4, g_conv_dim * 4, g_conv_dim * 4],
            "64": [g_conv_dim * 16, g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2],
            "128": [g_conv_dim * 16, g_conv_dim * 16, g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2],
            "256": [g_conv_dim * 16, g_conv_dim * 16, g_conv_dim * 8, g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2],
            "512": [g_conv_dim * 16, g_conv_dim * 16, g_conv_dim * 8, g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2, g_conv_dim]
        }

        g_out_dims_collection = {
            "32": [g_conv_dim * 4, g_conv_dim * 4, g_conv_dim * 4],
            "64": [g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2, g_conv_dim],
            "128": [g_conv_dim * 16, g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2, g_conv_dim],
            "256": [g_conv_dim * 16, g_conv_dim * 8, g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2, g_conv_dim],
            "512": [g_conv_dim * 16, g_conv_dim * 8, g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2, g_conv_dim, g_conv_dim]
        }

        bottom_collection = {"32": 4, "64": 4, "128": 4, "256": 4, "512": 4}

        self.z_dim = z_dim
        self.g_shared_dim = g_shared_dim
        self.num_classes = num_classes
        self.in_dims = g_in_dims_collection[str(img_size)]
        self.out_dims = g_out_dims_collection[str(img_size)]
        self.bottom = bottom_collection[str(img_size)]
        self.num_blocks = len(self.in_dims)
        self.chunk_size = z_dim // (self.num_blocks + 1)
        self.extra_feature_condition=extra_feature_condition
        
        self.affine_input_dim = self.chunk_size
        assert self.z_dim % (self.num_blocks + 1) == 0, "z_dim should be divided by the number of blocks"
        self.linear0 = MODULES.g_linear(in_features=self.chunk_size, out_features=self.in_dims[0]*self.bottom*self.bottom, bias=True)
        self.affine_input_dim += self.g_shared_dim

        if self.extra_feature_condition == True:
            self.affine_input_dim += self.g_shared_dim
            self.linear4extrafeature = MODULES.g_linear(in_features=4, out_features=self.g_shared_dim, bias=True)
        else:
            pass

        self.shared = ops.embedding(num_embeddings=self.num_classes, embedding_dim=self.g_shared_dim)
        self.blocks = []
        for index in range(self.num_blocks):
            self.blocks += [[
                GenBlock(in_channels=self.in_dims[index],
                         out_channels=self.out_dims[index],
                         affine_input_dim=self.affine_input_dim,
                         MODULES=MODULES)
            ]]

            if index + 1 in attn_g_loc and apply_attn:
                self.blocks += [[ops.SelfAttention(self.out_dims[index], is_generator=True, MODULES=MODULES)]]

        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

        self.bn4 = ops.batchnorm_2d(in_features=self.out_dims[-1])
        self.activation = MODULES.g_act_fn
        self.conv2d5 = MODULES.g_conv2d(in_channels=self.out_dims[-1], out_channels=1, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

        ops.init_weights(self.modules, 'ortho')

    def forward(self, z, label, node_feature=None):
        affine_list = []

        zs = torch.split(z, self.chunk_size, 1)
        z = zs[0]

        shared_label = self.shared(label)
        affine_list.append(shared_label)
        if len(affine_list) == 0:
            affines = [item for item in zs[1:]]
        else:
            if self.extra_feature_condition == True:
                extra_feature = self.linear4extrafeature(node_feature)
                affines = [torch.cat(affine_list + [item] + [extra_feature], 1) for item in zs[1:]]
            else:
                affines = [torch.cat(affine_list + [item], 1) for item in zs[1:]]

        act = self.linear0(z)
        act = act.view(-1, self.in_dims[0], self.bottom, self.bottom)
        counter = 0
        for index, blocklist in enumerate(self.blocks):
            for block in blocklist:
                if isinstance(block, ops.SelfAttention):
                    act = block(act)
                else:
                    act = block(act, affines[counter])
                    counter += 1

        act = self.bn4(act)
        act = self.activation(act)
        act = self.conv2d5(act)
        out = self.tanh(act)
        return out


class DiscOptBlock(nn.Module):
    def __init__(self, in_channels, out_channels, apply_d_sn, MODULES):
        super(DiscOptBlock, self).__init__()
        self.apply_d_sn = apply_d_sn

        self.conv2d0 = MODULES.d_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        self.conv2d1 = MODULES.d_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2d2 = MODULES.d_conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

        if not apply_d_sn:
            self.bn0 = MODULES.d_bn(in_features=in_channels)
            self.bn1 = MODULES.d_bn(in_features=out_channels)

        self.activation = MODULES.d_act_fn
        self.average_pooling = nn.AvgPool2d(2)

    def forward(self, x):
        x0 = x
        x = self.conv2d1(x)
        if not self.apply_d_sn:
            x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2d2(x)
        x = self.average_pooling(x)

        x0 = self.average_pooling(x0)
        if not self.apply_d_sn:
            x0 = self.bn0(x0)
        x0 = self.conv2d0(x0)
        out = x + x0
        return out


class DiscBlock(nn.Module):
    def __init__(self, in_channels, out_channels, apply_d_sn, MODULES, downsample=True):
        super(DiscBlock, self).__init__()
        self.apply_d_sn = apply_d_sn
        self.downsample = downsample

        self.activation = MODULES.d_act_fn

        self.ch_mismatch = False
        if in_channels != out_channels:
            self.ch_mismatch = True

        if self.ch_mismatch or downsample:
            self.conv2d0 = MODULES.d_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
            if not apply_d_sn:
                self.bn0 = MODULES.d_bn(in_features=in_channels)

        self.conv2d1 = MODULES.d_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2d2 = MODULES.d_conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

        if not apply_d_sn:
            self.bn1 = MODULES.d_bn(in_features=in_channels)
            self.bn2 = MODULES.d_bn(in_features=out_channels)

        self.average_pooling = nn.AvgPool2d(2)

    def forward(self, x):
        x0 = x
        if not self.apply_d_sn:
            x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2d1(x)

        if not self.apply_d_sn:
            x = self.bn2(x)
        x = self.activation(x)
        x = self.conv2d2(x)
        if self.downsample:
            x = self.average_pooling(x)

        if self.downsample or self.ch_mismatch:
            if not self.apply_d_sn:
                x0 = self.bn0(x0)
            x0 = self.conv2d0(x0)
            if self.downsample:
                x0 = self.average_pooling(x0)
        out = x + x0
        return out


class Discriminator(nn.Module):
    def __init__(self, img_size, d_conv_dim, apply_d_sn, apply_attn, attn_d_loc, num_classes, MODULES,
         extra_feature_condition=False):
        super(Discriminator, self).__init__()
        d_in_dims_collection = {
            "32": [1] + [d_conv_dim * 2, d_conv_dim * 2, d_conv_dim * 2],
            "64": [1] + [d_conv_dim, d_conv_dim * 2, d_conv_dim * 4, d_conv_dim * 8],
            "128": [1] + [d_conv_dim, d_conv_dim * 2, d_conv_dim * 4, d_conv_dim * 8, d_conv_dim * 16],
            "256": [1] + [d_conv_dim, d_conv_dim * 2, d_conv_dim * 4, d_conv_dim * 8, d_conv_dim * 8, d_conv_dim * 16],
            "512": [1] + [d_conv_dim, d_conv_dim, d_conv_dim * 2, d_conv_dim * 4, d_conv_dim * 8, d_conv_dim * 8, d_conv_dim * 16]
        }

        d_out_dims_collection = {
            "32": [d_conv_dim * 2, d_conv_dim * 2, d_conv_dim * 2, d_conv_dim * 2],
            "64": [d_conv_dim, d_conv_dim * 2, d_conv_dim * 4, d_conv_dim * 8, d_conv_dim * 16],
            "128": [d_conv_dim, d_conv_dim * 2, d_conv_dim * 4, d_conv_dim * 8, d_conv_dim * 16, d_conv_dim * 16],
            "256": [d_conv_dim, d_conv_dim * 2, d_conv_dim * 4, d_conv_dim * 8, d_conv_dim * 8, d_conv_dim * 16, d_conv_dim * 16],
            "512":
            [d_conv_dim, d_conv_dim, d_conv_dim * 2, d_conv_dim * 4, d_conv_dim * 8, d_conv_dim * 8, d_conv_dim * 16, d_conv_dim * 16]
        }

        d_down = {
            "32": [True, True, False, False],
            "64": [True, True, True, True, False],
            "128": [True, True, True, True, True, False],
            "256": [True, True, True, True, True, True, False],
            "512": [True, True, True, True, True, True, True, False]
        }

        self.num_classes = num_classes
        self.in_dims = d_in_dims_collection[str(img_size)]
        self.out_dims = d_out_dims_collection[str(img_size)]
        down = d_down[str(img_size)]

        self.extra_feature_condition = extra_feature_condition
        if self.extra_feature_condition == True:
            self.linear4extrafeature = MODULES.g_linear(in_features=4, out_features=self.out_dims[-1], bias=True)
        else:
            pass

        self.blocks = []
        for index in range(len(self.in_dims)):
            if index == 0:
                self.blocks += [[
                    DiscOptBlock(in_channels=self.in_dims[index], out_channels=self.out_dims[index], apply_d_sn=apply_d_sn, MODULES=MODULES)
                ]]
            else:
                self.blocks += [[
                    DiscBlock(in_channels=self.in_dims[index],
                              out_channels=self.out_dims[index],
                              apply_d_sn=apply_d_sn,
                              MODULES=MODULES,
                              downsample=down[index])
                ]]

            if index + 1 in attn_d_loc and apply_attn:
                self.blocks += [[ops.SelfAttention(self.out_dims[index], is_generator=False, MODULES=MODULES)]]

        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

        self.activation = MODULES.d_act_fn

        self.linear1 = MODULES.d_linear(in_features=self.out_dims[-1], out_features=1, bias=True)
        self.embedding = MODULES.d_embedding(num_classes, self.out_dims[-1])

        ops.init_weights(self.modules, 'ortho')

    def forward(self, x, label, node_feature=None):

        h = x
        for index, blocklist in enumerate(self.blocks):
            for block in blocklist:
                h = block(h)
        bottom_h, bottom_w = h.shape[2], h.shape[3]
        h = self.activation(h)
        h = torch.sum(h, dim=[2, 3])
        # adversarial training
        adv_output = torch.squeeze(self.linear1(h))
        adv_output = adv_output + torch.sum(torch.mul(self.embedding(label), h), 1)

        if self.extra_feature_condition == True:
            adv_output += torch.sum(torch.mul(self.linear4extrafeature(node_feature), h), 1)

        return adv_output