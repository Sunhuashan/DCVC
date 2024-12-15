# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from torch import nn

from src.layers.layers import conv3x3

from .common_model import CompressionModel
from .video_net import LowerBound, UNet, get_enc_dec_models, get_hyper_enc_dec_models
from ..utils.stream_helper import encode_i, decode_i, get_downsampled_shape, filesize, \
    get_rounded_q, get_state_dict

from .tcm import ConvTransBlock, ResidualBlockWithStride, ResidualBlockUpsample, \
    subpel_conv3x3

import logging
import torch
from typing import Union

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) 

file_handler = logging.FileHandler('mylog.log')
file_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter) 
logger.addHandler(file_handler)

class IntraNoAR(CompressionModel):
    def __init__(self, N=128, anchor_num=4, config=[2, 2, 2, 2, 2, 2], head_dim=[8, 16, 32, 32, 16, 8], drop_path_rate=0, M=320, num_slices=5, max_support_slices=5):
        super().__init__(y_distribution='gaussian', z_channel=192)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO) 

        file_handler = logging.FileHandler('mylog.log')
        file_handler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter) 
        self.logger.addHandler(file_handler)        

        self.z_channel = 192
        self.config = config
        self.head_dim = head_dim
        self.window_size = 8
        self.num_slices = num_slices
        self.max_support_slices = max_support_slices
        dim = N
        self.M = M
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(config))]
        begin = 0

        self.m_down1 = [ConvTransBlock(dim, dim, self.head_dim[0], self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW') 
                      for i in range(config[0])] + \
                      [ResidualBlockWithStride(2*N, 2*N, stride=2)]
        self.m_down2 = [ConvTransBlock(dim, dim, self.head_dim[1], self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW')
                      for i in range(config[1])] + \
                      [ResidualBlockWithStride(2*N, 2*N, stride=2)]
        self.m_down3 = [ConvTransBlock(dim, dim, self.head_dim[2], self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW')
                      for i in range(config[2])] + \
                      [conv3x3(2*N, M, stride=2)]

        self.m_up1 = [ConvTransBlock(dim, dim, self.head_dim[3], self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW') 
                      for i in range(config[3])] + \
                      [ResidualBlockUpsample(2*N, 2*N, 2)]
        self.m_up2 = [ConvTransBlock(dim, dim, self.head_dim[4], self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW') 
                      for i in range(config[4])] + \
                      [ResidualBlockUpsample(2*N, 2*N, 2)]
        self.m_up3 = [ConvTransBlock(dim, dim, self.head_dim[5], self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW') 
                      for i in range(config[5])] + \
                      [subpel_conv3x3(2*N, 3, 2)]
        
        self.enc = nn.Sequential(*[ResidualBlockWithStride(3, 2*N, 2)] + self.m_down1 + self.m_down2 + self.m_down3)
        self.dec = nn.Sequential(*[ResidualBlockUpsample(M, 2*N, 2)] + self.m_up1 + self.m_up2 + self.m_up3)

        # self.enc, self.dec = get_enc_dec_models(3, 16, N)
        self.refine = nn.Sequential(
            UNet(16, 16),
            conv3x3(16, 3),
        )
        self.hyper_enc, self.hyper_dec = get_hyper_enc_dec_models(M, self.z_channel)
        
        # self.y_prior_fusion = nn.Sequential(
        #     nn.Conv2d(self.z_channel * 2, self.z_channel * 3, 3, stride=1, padding=1),
        #     nn.LeakyReLU(0.2),
        #     nn.Conv2d(self.z_channel * 3, self.z_channel * 3, 3, stride=1, padding=1),
        #     nn.LeakyReLU(0.2),
        #     nn.Conv2d(self.z_channel * 3, self.z_channel * 3, 3, stride=1, padding=1)
        # )

        self.y_prior_fusion = nn.Sequential(
            nn.Conv2d(M * 2, M * 3, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(M * 3, M * 3, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(M * 3, M * 3, 3, stride=1, padding=1)
        )

        self.y_spatial_prior = nn.Sequential(
            nn.Conv2d(M * 4, M * 3, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(M * 3, M * 3, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(M * 3, M * 2, 3, padding=1)
        )

        self.q_basic = nn.Parameter(torch.ones((1, M, 1, 1)))
        self.q_scale = nn.Parameter(torch.ones((anchor_num, 1, 1, 1)))
        # the exact q_step is q_basic * q_scale
        self.N = int(N)
        self.anchor_num = int(anchor_num)

        # self._initialize_weights()

    def get_curr_q(self, q_scale):
        q_basic = LowerBound.apply(self.q_basic, 0.5)
        return q_basic * q_scale

    def forward(self, x, q_scale=None):
        ''' 
            x: {} torch.Size([1, 3, 512, 768])
            y: {} torch.Size([1, 192, 32, 48])
            z: {} torch.Size([1, 192, 8, 12])
            params: {} torch.Size([1, 384, 32, 48])
            q_step: {} torch.Size([1, 192, 32, 48])
            scales: {} torch.Size([1, 192, 32, 48])
            means: {} torch.Size([1, 192, 32, 48])
            y_res: {} torch.Size([1, 192, 32, 48])
            y_q: {} torch.Size([1, 192, 32, 48])
            y_hat: {} torch.Size([1, 192, 32, 48])
            scales_hat: {} torch.Size([1, 192, 32, 48])
        '''
        curr_q = self.get_curr_q(q_scale)

        # 3 -> 320
        # print("x: {}", x.shape)
        y = self.enc(x)
        y = y / curr_q
        # print("y: {}", y.shape)

        # 320 -> 192
        z = self.hyper_enc(y)
        z_hat = self.quant(z)
        # print("z: {}", z.shape)

        # 192 -> 2*320
        params = self.hyper_dec(z_hat)
        # print("params: {}", params.shape)

        # 2*320 -> 3*320
        q_step, scales, means = self.y_prior_fusion(params).chunk(3, 1)
        
        # print("q_step: {}", q_step.shape)
        # print("scales: {}", scales.shape)
        # print("means: {}", means.shape)

        self.logger.info(q_step)
        self.logger.info(scales)
        self.logger.info(means)

        # 320
        y_res, y_q, y_hat, scales_hat = self.forward_dual_prior(
            y, means, scales, q_step, self.y_spatial_prior)

        # print("y_res: {}", y_res.shape)
        # print("y_q: {}", y_q.shape)
        # print("y_hat: {}", y_hat.shape)
        # print("scales_hat: {}", scales_hat.shape)

        y_hat = y_hat * curr_q
        # self.logger.info(y_hat)
        x_hat = self.dec(y_hat)
        # x_hat = self.refine(y_hat)

        if self.training:
            # self.logger.info("Add Noise!")
            y_for_bit = self.add_noise(y_res)
            z_for_bit = self.add_noise(z)
        else:
            y_for_bit = y_q
            z_for_bit = z_hat

        # print(y_for_bit)
        # print(scales_hat)
        bits_y = self.get_y_gaussian_bits(y_for_bit, scales_hat)
        bits_z = self.get_z_bits(z_for_bit, self.bit_estimator_z)
        mse = self.mse(x, x_hat)
        ssim = self.ssim(x, x_hat)

        _, _, H, W = x.size()
        pixel_num = H * W
        bpp_y = torch.sum(bits_y, dim=(1, 2, 3)) / pixel_num
        bpp_z = torch.sum(bits_z, dim=(1, 2, 3)) / pixel_num
        mse = torch.sum(mse, dim=(1, 2, 3)) / pixel_num

        bits = torch.sum(bpp_y + bpp_z) * pixel_num
        bpp = bpp_y + bpp_z

        # print({
        #     "x_hat": x_hat,
        #     "mse": mse,
        #     "ssim": ssim,
        #     "bit": bits.item(),
        #     "bpp": bpp,
        #     "bpp_y": bpp_y,
        #     "bpp_z": bpp_z,
        # })

        return {
            "x_hat": x_hat,
            "mse": mse,
            "ssim": ssim,
            "bit": bits.item(),
            "bpp": bpp,
            "bpp_y": bpp_y,
            "bpp_z": bpp_z,
        }

    @staticmethod
    def get_q_scales_from_ckpt(ckpt_path):
        ckpt = get_state_dict(ckpt_path)
        q_scales = ckpt["q_scale"]
        return q_scales.reshape(-1)

    def encode_decode(self, x, q_scale, output_path=None, pic_width=None, pic_height=None):
        # pic_width and pic_height may be different from x's size. X here is after padding
        # x_hat has the same size with x
        if output_path is None:
            return self.forward(x, q_scale)

        assert pic_height is not None
        assert pic_width is not None
        q_scale, q_index = get_rounded_q(q_scale)
        compressed = self.compress(x, q_scale)
        bit_stream = compressed['bit_stream']
        encode_i(pic_height, pic_width, q_index, bit_stream, output_path)
        bit = filesize(output_path) * 8

        height, width, q_index, bit_stream = decode_i(output_path)
        decompressed = self.decompress(bit_stream, height, width, q_index / 100)
        x_hat = decompressed['x_hat']

        result = {
            'bit': bit,
            'x_hat': x_hat,
        }
        return result

    def compress(self, x, q_scale):
        curr_q = self.get_curr_q(q_scale)

        y = self.enc(x)
        y = y / curr_q
        z = self.hyper_enc(y)
        z_hat = torch.round(z)

        params = self.hyper_dec(z_hat)
        q_step, scales, means = self.y_prior_fusion(params).chunk(3, 1)
        y_q_w_0, y_q_w_1, scales_w_0, scales_w_1, y_hat = self.compress_dual_prior(
            y, means, scales, q_step, self.y_spatial_prior)
        y_hat = y_hat * curr_q

        self.entropy_coder.reset_encoder()
        _ = self.bit_estimator_z.encode(z_hat)
        _ = self.gaussian_encoder.encode(y_q_w_0, scales_w_0)
        _ = self.gaussian_encoder.encode(y_q_w_1, scales_w_1)
        bit_stream = self.entropy_coder.flush_encoder()

        result = {
            "bit_stream": bit_stream,
        }
        return result

    def decompress(self, bit_stream, height, width, q_scale):
        curr_q = self.get_curr_q(q_scale)

        self.entropy_coder.set_stream(bit_stream)
        device = next(self.parameters()).device
        z_size = get_downsampled_shape(height, width, 64)
        z_hat = self.bit_estimator_z.decode_stream(z_size)
        z_hat = z_hat.to(device)

        params = self.hyper_dec(z_hat)
        q_step, scales, means = self.y_prior_fusion(params).chunk(3, 1)
        y_hat = self.decompress_dual_prior(means, scales, q_step, self.y_spatial_prior)

        y_hat = y_hat * curr_q
        x_hat = self.refine(self.dec(y_hat)).clamp_(0, 1)
        return {"x_hat": x_hat}
