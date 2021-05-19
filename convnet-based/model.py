import torch
import torch.nn as nn
from torch.nn.modules.upsampling import Upsample

'''
    Example model construction in pytorch
'''
class example_resblock(nn.Module):
    def __init__(self, bias=True, act=nn.ReLU(True)):
        super(example_resblock, self).__init__()
        modules = []
        modules.append(nn.Conv2d(16, 16, 3, padding=1), bias=bias)
        modules.append(act)
        modules.append(nn.Conv2d(16, 16, 3, padding=1), bias=bias)
        self.body = nn.Sequential(*modules)

    def forward(self, x):
        out = self.body(x)
        out += x
        return out

class resblock(nn.Module):
    def __init__(self, nFeat, kernel_size=3, bias=True, act=nn.ReLU(True)):
        super(resblock, self).__init__()
        modules = []
        modules.append(nn.Conv2d(nFeat, nFeat, kernel_size=kernel_size, padding=kernel_size // 2, bias=bias))
        modules.append(act)
        modules.append(nn.Conv2d(nFeat, nFeat, kernel_size=kernel_size, padding=kernel_size // 2, bias=bias))
        self.body = nn.Sequential(*modules)
    
    def forward(self, x):
        out = self.body(x)
        out += x
        return out

class upsampler(nn.Module):
    def __init__(self, scale=2, nFeat=16, act=nn.ReLU(True)):
        super(upsampler, self).__init__()
        #===== write your model definition here =====#
        modules = []
        modules.append(nn.Conv2d(nFeat, nFeat*4, kernel_size=3, padding=3 // 2, bias=True))
        modules.append(nn.PixelShuffle(scale))
        modules.append(act)
        self.body = nn.Sequential(*modules)

    def forward(self, x):
        #===== write your dataflow here =====#
        out = self.body(x)
        return out

class ZebraSRNet(nn.Module):
    def __init__(self, nFeat=64, kernel_size=3, nResBlock=8, imgChannel=3):
        super(ZebraSRNet, self).__init__()
        #===== write your model definition here using 'resblock' and 'upsampler' as the building blocks =====#
        modules1 = []
        modules1.append(nn.Conv2d(kernel_size, nFeat, kernel_size, padding= kernel_size//2, bias=True))
        self.body1 = nn.Sequential(*modules1)

        modules2 = []
        for i in range(0, nResBlock, 1):
            modules2.append(resblock(nFeat, kernel_size, True, nn.ReLU(True)))
        self.body2 = nn.Sequential(*modules2)

        modules3 = []
        modules3.append(upsampler(2, nFeat, act=nn.ReLU(True)))
        modules3.append(upsampler(2, nFeat, act=nn.ReLU(True)))
        modules3.append(nn.Conv2d(nFeat, kernel_size, kernel_size, padding= kernel_size//2, bias=True))
        self.body3 = nn.Sequential(*modules3)

    def forward(self, x):
        #===== write your dataflow here =====#
        out1 = self.body1(x)

        out2 = self.body2(out1)
        out2 += out1
        
        out3 = self.body3(out2)
        return out3