# Implementation of the filters is borrowed from paper "Alias-Free Generative Adversarial Networks (StyleGAN3)" https://nvlabs.github.io/stylegan3/
# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


"""Convolutional neural operator."""

from torch import nn
import torch


class CNOBlock3d(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        in_size,
        out_size,
        cutoff_den=2.0001,
        conv_kernel=3,
        filter_size=6,
        lrelu_upsampling=2,
        half_width_mult=0.8,
        radial=False,
        batch_norm=True,
        activation="cno_lrelu",
    ):
        super(CNOBlock3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_size = in_size
        self.out_size = out_size
        self.conv_kernel = conv_kernel
        self.batch_norm = batch_norm
        self.citically_sampled = False
        if cutoff_den == 2.0:
            self.citically_sampled = True
        self.in_cutoff = self.in_size / cutoff_den
        self.out_cutoff = self.out_size / cutoff_den
        self.in_halfwidth = half_width_mult * self.in_size - self.in_size / cutoff_den
        self.out_halfwidth = (
            half_width_mult * self.out_size - self.out_size / cutoff_den
        )
        pad = (self.conv_kernel - 1) // 2
        self.convolution = torch.nn.Conv3d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.conv_kernel,
            padding=pad,
        )
        if self.batch_norm:
            self.batch_norm = nn.BatchNorm3d(self.out_channels)
        if activation == "LeakyReLU":
            self.activation = nn.LeakyReLU(negative_slope=0.2)
        else:
            raise ValueError(f"Activation function {activation} not supported")

    def forward(self, x):
        x = self.convolution(x)
        if self.batch_norm:
            x = self.batch_norm(x)
        return self.activation(x)


class LiftProjectBlock3d(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        in_size,
        out_size,
        latent_dim=64,
        cutoff_den=2.0001,
        conv_kernel=3,
        filter_size=6,
        lrelu_upsampling=2,
        half_width_mult=0.8,
        radial=False,
        batch_norm=True,
        activation="cno_lrelu",
    ):
        super(LiftProjectBlock3d, self).__init__()
        self.inter_CNOBlock = CNOBlock3d(
            in_channels=in_channels,
            out_channels=latent_dim,
            in_size=in_size,
            out_size=out_size,
            cutoff_den=cutoff_den,
            conv_kernel=conv_kernel,
            filter_size=filter_size,
            lrelu_upsampling=lrelu_upsampling,
            half_width_mult=half_width_mult,
            radial=radial,
            batch_norm=batch_norm,
            activation=activation,
        )
        pad = (conv_kernel - 1) // 2
        self.convolution = torch.nn.Conv3d(
            in_channels=latent_dim,
            out_channels=out_channels,
            kernel_size=conv_kernel,
            stride=1,
            padding=pad,
        )
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.batch_norm = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = self.inter_CNOBlock(x)
        x = self.convolution(x)
        if self.batch_norm:
            x = self.batch_norm(x)
        return x


class ResidualBlock3d(nn.Module):

    def __init__(
        self,
        channels,
        size,
        cutoff_den=2.0001,
        conv_kernel=3,
        filter_size=6,
        lrelu_upsampling=2,
        half_width_mult=0.8,
        radial=False,
        batch_norm=True,
        activation="cno_lrelu",
    ):
        super(ResidualBlock3d, self).__init__()
        self.channels = channels
        self.size = size
        self.conv_kernel = conv_kernel
        self.batch_norm = batch_norm
        self.citically_sampled = False
        if cutoff_den == 2.0:
            self.citically_sampled = True
        self.cutoff = self.size / cutoff_den
        self.halfwidth = half_width_mult * self.size - self.size / cutoff_den
        pad = (self.conv_kernel - 1) // 2
        self.convolution1 = torch.nn.Conv3d(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=self.conv_kernel,
            stride=1,
            padding=pad,
        )
        self.convolution2 = torch.nn.Conv3d(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=self.conv_kernel,
            stride=1,
            padding=pad,
        )
        if self.batch_norm:
            self.batch_norm1 = nn.BatchNorm3d(self.channels)
            self.batch_norm2 = nn.BatchNorm3d(self.channels)
        if activation == "LeakyReLU":
            self.activation = nn.LeakyReLU(negative_slope=0.2)
        else:
            raise ValueError(f"Activation function {activation} not supported")

    def forward(self, x):
        out = self.convolution1(x)
        if self.batch_norm:
            out = self.batch_norm1(out)
        out = self.activation(out)
        out = self.convolution2(out)
        if self.batch_norm:
            out = self.batch_norm2(out)
        return x + out


class CNO3d(nn.Module):

    def __init__(
        self,
        in_dim,
        in_size,
        N_layers,
        N_res=1,
        N_res_neck=6,
        channel_multiplier=32,
        conv_kernel=3,
        cutoff_den=2.0001,
        filter_size=6,
        lrelu_upsampling=2,
        half_width_mult=0.8,
        radial=False,
        batch_norm=True,
        out_dim=1,
        out_dim_mult=1,
        out_size=1,
        expand_input=False,
        latent_lift_proj_dim=64,
        add_inv=True,
        activation="LeakyReLU",
    ):
        super(CNO3d, self).__init__()
        self.N_layers = int(N_layers)
        self.lift_dim = channel_multiplier // 2
        self.out_dim_mult = out_dim_mult
        self.out_dim = out_dim * out_dim_mult
        self.add_inv = add_inv
        self.channel_multiplier = channel_multiplier
        if radial == 0:
            self.radial = False
        else:
            self.radial = True
        self.encoder_features = [self.lift_dim]
        for i in range(self.N_layers):
            self.encoder_features.append(2**i * self.channel_multiplier)
        self.decoder_features_in = self.encoder_features[1:]
        self.decoder_features_in.reverse()
        self.decoder_features_out = self.encoder_features[:-1]
        self.decoder_features_out.reverse()
        for i in range(1, self.N_layers):
            self.decoder_features_in[i] = 2 * self.decoder_features_in[i]
        self.inv_features = self.decoder_features_in
        self.inv_features.append(
            self.encoder_features[0] + self.decoder_features_out[-1]
        )
        if not expand_input:
            latent_size = in_size
        else:
            down_exponent = 2**N_layers
            latent_size = in_size - in_size % down_exponent + down_exponent
        if out_size == 1:
            latent_size_out = latent_size
        elif not expand_input:
            latent_size_out = out_size
        else:
            down_exponent = 2**N_layers
            latent_size_out = out_size - out_size % down_exponent + down_exponent
        self.encoder_sizes = []
        self.decoder_sizes = []
        for i in range(self.N_layers + 1):
            self.encoder_sizes.append(latent_size // 2**i)
            self.decoder_sizes.append(latent_size_out // 2 ** (self.N_layers - i))
        self.lift = LiftProjectBlock3d(
            in_channels=in_dim,
            out_channels=self.encoder_features[0],
            in_size=in_size,
            out_size=self.encoder_sizes[0],
            latent_dim=latent_lift_proj_dim,
            cutoff_den=cutoff_den,
            conv_kernel=conv_kernel,
            filter_size=filter_size,
            lrelu_upsampling=lrelu_upsampling,
            half_width_mult=half_width_mult,
            radial=radial,
            batch_norm=False,
            activation=activation,
        )
        _out_size = out_size
        if out_size == 1:
            _out_size = in_size
        self.project = LiftProjectBlock3d(
            in_channels=self.encoder_features[0] + self.decoder_features_out[-1],
            out_channels=self.out_dim,
            in_size=self.decoder_sizes[-1],
            out_size=_out_size,
            latent_dim=latent_lift_proj_dim,
            cutoff_den=cutoff_den,
            conv_kernel=conv_kernel,
            filter_size=filter_size,
            lrelu_upsampling=lrelu_upsampling,
            half_width_mult=half_width_mult,
            radial=radial,
            batch_norm=False,
            activation=activation,
        )
        self.encoder = nn.ModuleList(
            [
                CNOBlock3d(
                    in_channels=self.encoder_features[i],
                    out_channels=self.encoder_features[i + 1],
                    in_size=self.encoder_sizes[i],
                    out_size=self.encoder_sizes[i + 1],
                    cutoff_den=cutoff_den,
                    conv_kernel=conv_kernel,
                    filter_size=filter_size,
                    lrelu_upsampling=lrelu_upsampling,
                    half_width_mult=half_width_mult,
                    radial=radial,
                    batch_norm=batch_norm,
                    activation=activation,
                )
                for i in range(self.N_layers)
            ]
        )
        self.ED_expansion = nn.ModuleList(
            [
                CNOBlock3d(
                    in_channels=self.encoder_features[i],
                    out_channels=self.encoder_features[i],
                    in_size=self.encoder_sizes[i],
                    out_size=self.decoder_sizes[self.N_layers - i],
                    cutoff_den=cutoff_den,
                    conv_kernel=conv_kernel,
                    filter_size=filter_size,
                    lrelu_upsampling=lrelu_upsampling,
                    half_width_mult=half_width_mult,
                    radial=radial,
                    batch_norm=batch_norm,
                    activation=activation,
                )
                for i in range(self.N_layers + 1)
            ]
        )
        self.decoder = nn.ModuleList(
            [
                CNOBlock3d(
                    in_channels=self.decoder_features_in[i],
                    out_channels=self.decoder_features_out[i],
                    in_size=self.decoder_sizes[i],
                    out_size=self.decoder_sizes[i + 1],
                    cutoff_den=cutoff_den,
                    conv_kernel=conv_kernel,
                    filter_size=filter_size,
                    lrelu_upsampling=lrelu_upsampling,
                    half_width_mult=half_width_mult,
                    radial=radial,
                    batch_norm=batch_norm,
                    activation=activation,
                )
                for i in range(self.N_layers)
            ]
        )
        self.decoder_inv = nn.ModuleList(
            [
                CNOBlock3d(
                    in_channels=self.inv_features[i],
                    out_channels=self.inv_features[i],
                    in_size=self.decoder_sizes[i],
                    out_size=self.decoder_sizes[i],
                    cutoff_den=cutoff_den,
                    conv_kernel=conv_kernel,
                    filter_size=filter_size,
                    lrelu_upsampling=lrelu_upsampling,
                    half_width_mult=half_width_mult,
                    radial=radial,
                    batch_norm=batch_norm,
                    activation=activation,
                )
                for i in range(self.N_layers + 1)
            ]
        )
        self.res_nets = []
        self.N_res = int(N_res)
        self.N_res_neck = int(N_res_neck)
        for l in range(self.N_layers):
            for i in range(self.N_res):
                self.res_nets.append(
                    ResidualBlock3d(
                        channels=self.encoder_features[l],
                        size=self.encoder_sizes[l],
                        cutoff_den=cutoff_den,
                        conv_kernel=conv_kernel,
                        filter_size=filter_size,
                        lrelu_upsampling=lrelu_upsampling,
                        half_width_mult=half_width_mult,
                        radial=radial,
                        batch_norm=batch_norm,
                        activation=activation,
                    )
                )
        for i in range(self.N_res_neck):
            self.res_nets.append(
                ResidualBlock3d(
                    channels=self.encoder_features[self.N_layers],
                    size=self.encoder_sizes[self.N_layers],
                    cutoff_den=cutoff_den,
                    conv_kernel=conv_kernel,
                    filter_size=filter_size,
                    lrelu_upsampling=lrelu_upsampling,
                    half_width_mult=half_width_mult,
                    radial=radial,
                    batch_norm=batch_norm,
                    activation=activation,
                )
            )
        self.res_nets = torch.nn.Sequential(*self.res_nets)

    def forward(self, x):
        switch = False
        if x.dim() == 5 and x.shape[-1] < x.shape[1]:
            switch = True
            x = x.permute(0, 4, 1, 2, 3)
        x = self.lift(x)
        skip = []
        for i in range(self.N_layers):
            y = x
            for j in range(self.N_res):
                y = self.res_nets[i * self.N_res + j](y)
            skip.append(y)
            x = self.encoder[i](x)
        for j in range(self.N_res_neck):
            x = self.res_nets[-j - 1](x)
        for i in range(self.N_layers):
            if i == 0:
                x = self.ED_expansion[self.N_layers - i](x)
            else:
                x = torch.cat((x, self.ED_expansion[self.N_layers - i](skip[-i])), 1)
            if self.add_inv:
                x = self.decoder_inv[i](x)
            x = self.decoder[i](x)
        x = torch.cat((x, self.ED_expansion[0](skip[0])), 1)
        x = self.project(x)
        del skip
        del y
        if switch:
            x = x.permute(0, 2, 3, 4, 1)
        if self.out_dim_mult > 1:
            x = x.reshape(
                x.shape[0],
                -1,
                x.shape[2],
                x.shape[3],
                self.out_dim // self.out_dim_mult,
            )
        return x
