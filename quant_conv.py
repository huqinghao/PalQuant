import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

__all__ = ['QConv','QConv1x1','QConv3x3']

class STE_discretizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_in, num_levels):
        x = x_in * (num_levels - 1)
        x = torch.round(x)
        x_out = x / (num_levels - 1)
        return x_out
    @staticmethod
    def backward(ctx, g):
        return g, None

# ref. https://github.com/ricky40403/DSQ/blob/master/DSQConv.py#L18
class QConv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, args, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(QConv, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.quan_weight = args.QWeightFlag
        self.quan_act = args.QActFlag
        self.STE_discretizer = STE_discretizer.apply
    
        if self.quan_weight:
            self.weight_levels = args.weight_levels
            self.uW = nn.Parameter(data = torch.tensor(0).float())
            self.lW = nn.Parameter(data = torch.tensor(0).float())
            self.register_buffer('bkwd_scaling_factorW', torch.tensor(args.bkwd_scaling_factorW).float())

        if self.quan_act:
            self.act_levels = args.act_levels
            self.uA = nn.Parameter(data = torch.tensor(0).float())
            self.lA = nn.Parameter(data = torch.tensor(0).float())
            self.register_buffer('bkwd_scaling_factorA', torch.tensor(args.bkwd_scaling_factorA).float())

        self.register_buffer('init', torch.tensor([0]))
        self.output_scale = nn.Parameter(data = torch.tensor(1).float())
        
        self.hook_Qvalues = False
        self.buff_weight = None
        self.buff_act = None

    def weight_quantization(self, weight):
        weight = (weight - self.lW) / (self.uW - self.lW)
        weight = weight.clamp(min=0, max=1) # [0, 1]

        weight = self.STE_discretizer(weight, self.weight_levels)
            
        weight = (weight - 0.5) * 2 # [-1, 1]

        return weight

    def act_quantization(self, x):
        x = (x - self.lA) / (self.uA - self.lA)
        x = x.clamp(min=0, max=1) # [0, 1]
        x = self.STE_discretizer(x, self.act_levels)

        return x

    def initialize(self, x):
        # self.init.data.fill_(0)
        Qweight = self.weight
        Qact = x
        
        if self.quan_weight:
            self.uW.data.fill_(self.weight.std()*3.0)
            self.lW.data.fill_(-self.weight.std()*3.0)
            Qweight = self.weight_quantization(self.weight)

        if self.quan_act:
            self.uA.data.fill_(x.std() / math.sqrt(1 - 2/math.pi) * 3.0)
            self.lA.data.fill_(x.min())
            Qact = self.act_quantization(x)

        Qout = F.conv2d(Qact, Qweight, self.bias,  self.stride, self.padding, self.dilation, self.groups)
        out = F.conv2d(x, self.weight, self.bias,  self.stride, self.padding, self.dilation, self.groups)
        self.output_scale.data.fill_(out.abs().mean() / Qout.abs().mean())


    def forward(self, x):
        if self.init == 1:
            self.initialize(x)
        
        Qweight = self.weight
        # if self.quan_weight:
        Qweight = self.weight_quantization(Qweight)

        Qact = x
        # if self.quan_act:
        Qact = self.act_quantization(Qact)

        output = F.conv2d(Qact, Qweight, self.bias,  self.stride, self.padding, self.dilation, self.groups) * torch.abs(self.output_scale)

        return output

def QConv3x3(args, in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return QConv(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False, dilation=dilation, args=args)

def QConv1x1(args, in_planes, out_planes, stride=1,groups=1):
    """1x1 convolution"""
    return QConv(in_planes, out_planes, kernel_size=1, stride=stride,groups=groups, bias=False, args=args)
