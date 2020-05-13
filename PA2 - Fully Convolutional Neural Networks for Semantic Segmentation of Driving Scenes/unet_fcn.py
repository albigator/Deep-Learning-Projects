import torch.nn as nn
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np


def center_crop(x, comp_tensor):
    flagh = 0
    flagw = 0
    padh = (x.shape[-2] - comp_tensor.shape[-2])
    if padh % 2 == 1:
        flagh = 1
    padh = int(padh/2)
    padw = (x.shape[-1] - comp_tensor.shape[-1])
    if padw % 2 == 1:
        flagw = 1
    padw = int(padw/2)
    return F.pad(x,(-padw-flagw,-padw,-padh-flagh,-padh))

def conv_block(input_chans, output_chans):
    '''
    recurrent architecture: conv, relu, conv, relu.
    this will be cloned 7 times. 3 times going down, 3 times going up, once in end.
    i.e. 3 will have maxpool, 3 will have upsampling, 1 will have classification.
    '''
    return nn.Sequential(
        nn.Conv2d(input_chans, output_chans, kernel_size=3, stride=1, padding=0, dilation=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(output_chans, output_chans, kernel_size=3, stride=1, padding=0, dilation=1),
        nn.ReLU(inplace=True)
    )


class uNet(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        self.convdown1 = conv_block(3,64)
        self.convdown2 = conv_block(64,128)
        self.convdown3 = conv_block(128,256)
        self.convdown4 = conv_block(256,512)
        self.maxpool   = nn.MaxPool2d(2)
        self.upsampler = nn.Upsample(scale_factor=2, mode='bicubic')
        self.convup3   = conv_block(256 + 512, 256)
        self.convup2   = conv_block(128 + 256, 128)
        self.convup1   = conv_block(64  + 128, 64)
        self.classifier= nn.Conv2d(64,n_class,kernel_size=3, stride=1, padding=0, dilation=1)
       
       


    def forward(self, x, y):
        #go through downconvs,storing the 3 forward props
        send1 = self.convdown1(x)
        send2 = self.convdown2(self.maxpool(send1))
        send3 = self.convdown3(self.maxpool(send2))
        #conv, then upsample, then combine with the forwardpropped output
        x     = self.convdown4(self.maxpool(send3))
        x     = self.upsampler(x)
        #center crop the feedforward image
        send3 = center_crop(send3,x)
        x     = torch.cat([x,send3],dim = 1)
        x     = self.upsampler(self.convup3(x))
        send2 = center_crop(send2,x)
        x     = torch.cat([x,send2],dim  = 1)
        x     = self.upsampler(self.convup2(x))
        send1 = center_crop(send1,x)
        x     = torch.cat([x,send1],dim = 1)
        #last conv
        out     = self.classifier(self.convup1(x))
        return out, center_crop(y,out)
        
