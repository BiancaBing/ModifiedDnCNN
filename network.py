import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import initialize_weights


color = 3


class BottleNeck(nn.Module):
    """docstring for BottleNeck"""

    def __init__(self, inplanes, planes, stride=1):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, inplanes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(inplanes)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.leaky_relu(out, 0.1)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.leaky_relu(out, 0.1)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = F.leaky_relu(out, 0.1)

        return out

class MemoryBlock(nn.Module):
    """Note: num_memblock denotes the number of MemoryBlock currently"""

    def __init__(self, channels, num_resblock, num_memblock):
        super(MemoryBlock, self).__init__()
        self.recursive_unit = nn.ModuleList(
            [ResidualBlock(channels) for i in range(num_resblock)]
        )
        self.gate_unit = GateUnit((num_resblock + num_memblock) * channels, channels, True)  # kernel 1x1

    def forward(self, x, ys):
        """ys is a list which contains long-term memory coming from previous memory block
        xs denotes the short-term memory coming from recursive unit
        """
        xs = []
        residual = x
        for layer in self.recursive_unit:
            x = layer(x)
            xs.append(x)

        # gate_out = self.gate_unit(torch.cat([xs,ys], dim=1))
        gate_out = self.gate_unit(torch.cat(xs + ys, 1))  # where xs and ys are list, so concat operation is xs+ys
        ys.append(gate_out)
        return gate_out


class ResidualBlock(torch.nn.Module):
    """ResidualBlock
    x - Relu - Conv - Relu - Conv - x
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.relu_conv1 = BNReLUConv(channels, channels, True)
        self.relu_conv2 = BNReLUConv(channels, channels, True)

    def forward(self, x):
        residual = x
        out = self.relu_conv1(x)
        out = self.relu_conv2(out)
        out = out + residual
        return out

class BNReLUConv(nn.Sequential):
    def __init__(self, in_channels, channels, inplace=True):
        super(BNReLUConv, self).__init__()
        self.add_module('bn', nn.BatchNorm2d(in_channels))
        self.add_module('relu',
                        nn.ReLU(inplace=inplace))  # tureL: direct modified x, false: new object and the modified
        self.add_module('conv',
                        nn.Conv2d(in_channels, channels, 3, 1, 1))  # bias: defautl: ture on pytorch, learnable bias


class GateUnit(nn.Sequential):
    def __init__(self, in_channels, channels, inplace=True):
        super(GateUnit, self).__init__()
        self.add_module('bn', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=inplace))
        self.add_module('conv', nn.Conv2d(in_channels, channels, 1, 1, 0))

# Based on the structure of DnCNN and MemNET
# Add Bottleneck layer
class ModifiedDnCNN(nn.Module):
    def __init__(self, depth=10, n_channels=64, image_channels=3, use_bnorm=True, kernel_size=3,num_memblock=3, num_resblock=1):
        super(ModifiedDnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []

        layers.append(
            nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                      bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth - 2):
            layers.append(
                nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                          bias=True))
            layers.append(nn.ReLU(inplace=True))
            layers.append(BottleNeck(n_channels, n_channels))
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()
        self.reconstructor = BNReLUConv(n_channels, image_channels, True)
        self.dense_memory = nn.ModuleList(
            [MemoryBlock(n_channels, num_resblock, i + 1) for i in range(num_memblock)]
        )
        self.weights = nn.Parameter((torch.ones(1, num_memblock) / num_memblock), requires_grad=True)

    def forward(self, x):
        residual = x
        out = self.dncnn(x)
        w_sum = self.weights.sum(1)
        mid_feat = []  # A lsit contains the output of each memblock
        ys = [out]  # A list contains previous memblock output(long-term memory)  and the output of FENet
        for memory_block in self.dense_memory:
            out = memory_block(out, ys)  # out is the output of GateUnit  channels=64
            mid_feat.append(out);
        # pred = Variable(torch.zeros(x.shape).type(dtype),requires_grad=False)
        pred = (self.reconstructor(mid_feat[0]) + residual) * self.weights.data[0][0] / w_sum
        for i in range(1, len(mid_feat)):
            pred = pred + (self.reconstructor(mid_feat[i]) + residual) * self.weights.data[0][i] / w_sum

        return pred

    def _initialize_weights(self):
        for m in self.modules():
            initialize_weights(m)
