# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os

import numpy as np
from PIL import Image


class PNGReader():
    def __init__(self, src_folder, width, height):
        self.src_folder = src_folder
        # pngs = os.listdir(self.src_folder)
        self.width = width
        self.height = height
        # if 'im1.png' in pngs:
        #     self.padding = 1
        # elif 'im00001.png' in pngs:
        #     self.padding = 5
        # else:
        #   raise ValueError('unknown image naming convention; please specify')
        self.png_files = []
        for file in os.listdir(self.src_folder):
            # print(src_folder)
            if file[-3:] in ["jpg", "png", "peg"]:
                self.png_files.append(file)
        
        self.current_frame_index = 0
        self.eof = False

    def read_one_frame(self, src_format="rgb"):
        def _none_exist_frame():
            if src_format == "rgb":
                return None
            return None, None, None
        if self.eof:
            return _none_exist_frame()

        # png_path = os.path.join(self.src_folder,
        #                         f"im{str(self.current_frame_index).zfill(self.padding)}.png"
        #                         )
        png_file = self.png_files[self.current_frame_index]
        png_path = os.path.join(self.src_folder, png_file)
        if not os.path.exists(png_path):
            self.eof = True
            return _none_exist_frame()

        rgb = Image.open(png_path).convert('RGB')
        rgb = np.asarray(rgb).astype('float32').transpose(2, 0, 1)
        rgb = rgb / 255.
        _, height, width = rgb.shape
        # assert height == self.height
        # assert width == self.width

        self.current_frame_index += 1

        if self.current_frame_index >= len(self.png_files):  # 如果读取到最后一帧
            self.eof = True
        
        return rgb

    def close(self):
        self.current_frame_index = 0
