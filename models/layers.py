import torch
import torch.nn as nn
from  torch.nn.modules.conv import _pair, _ConvNd, Conv2d
import torch.nn.functional as F

class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        # self.conv2d = Conv2dF.apply
        # self.out = in_channels

    def forward(self, input):
        # temp = input.data
        # var, mean = torch.var_mean(temp, unbiased=False)
        input = input.sub(input.mean()).div(torch.sqrt(input.var(unbiased=False).add(1e-5)))
        return F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class AddBias(nn.Module):
    def __init__(self, num_features, threshold=8):
        super(AddBias, self).__init__()
        self.num_features = num_features
        self.threshold = threshold
        self.register_buffer('count', torch.tensor(0))
    
    def forward(self, x):
        if self.count < self.threshold:
            mu = x.mean(dim=self.dim, keepdim=True).detach()
            self.init_mean += (mu - self.init_mean) / (self.count + 1)
            self.count += 1
            return x.add(self._bias.mul(1e-2).sub(self.init_mean))
        return x.add(self._bias.sub(self.init_mean))

    def extra_repr(self):
        return 'num_features={}, threshold={}'.format(
            self.num_features, self.threshold)

class Bias1D(AddBias):
    def __init__(self, num_features, threshold=8):
        super(Bias1D, self).__init__(num_features, threshold)
        self._bias = nn.Parameter(torch.zeros(1, num_features))
        self.register_buffer('init_mean', torch.zeros(1, num_features))
        self.dim = 0
    
class Bias2D(AddBias):
    def __init__(self, num_features, threshold=8):
        super(Bias2D, self).__init__(num_features, threshold)
        self._bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.register_buffer('init_mean', torch.zeros(1, num_features, 1, 1))
        self.dim = (0,2,3)
    
class Bias3D(AddBias):
    def __init__(self, num_features, threshold=8):
        super(Bias3D, self).__init__(num_features, threshold)
        self._bias = nn.Parameter(torch.zeros(1, num_features, 1, 1, 1))
        self.register_buffer('init_mean', torch.zeros(1, num_features, 1, 1, 1))
        self.dim = (0,2,3,4)