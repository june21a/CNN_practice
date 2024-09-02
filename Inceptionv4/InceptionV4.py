import torch
from torch import nn


def conv_block(in_channel, out_channel, filter_size, stride, padding):
    block = []
    block.append(nn.Conv2d(in_channel, out_channel, filter_size, stride = stride, padding=padding))
    block.append(nn.BatchNorm2d(out_channel))
    block.append(nn.ReLU())
    return block


class InceptionV4StemBlock(nn.Module):
    def __init__(self):
        super(InceptionV4StemBlock ,self).__init__()
        
        
        self.conv1 = nn.Sequential(
            *(conv_block(3, 32, (3, 3), stride = 2, padding=0) +
            conv_block(32, 32, (3, 3), stride=1, padding=0) +
            conv_block(32, 64, (3, 3), stride=1, padding=1))
        )
        
        
        self.maxpool2_1 = nn.MaxPool2d((3, 3), stride=2, padding=0)
        self.conv2_1 = nn.Sequential(*conv_block(64, 96, (3, 3), stride=2, padding=0))
        
        
        self.conv3_l = nn.Sequential(
            *(conv_block(160, 64, (1, 1), stride = 1, padding=0) +
            conv_block(64, 96, (3, 3), stride = 1, padding=0))
        )
        self.conv3_r = nn.Sequential(
            *(conv_block(160, 64, (1, 1), stride = 1, padding=1) +
            conv_block(64, 64, (7, 1), stride = 1, padding=1) +
            conv_block(64, 64, (1, 7), stride = 1, padding=1) +
            conv_block(64, 96, (3, 3), stride = 1, padding=0))
        )
        
        
        self.conv4_1 = nn.Sequential(*conv_block(192, 192, (3, 3), stride = 2, padding=0))
        self.maxpool4_1 = nn.MaxPool2d((3, 3), stride=2, padding=0)
    
    def forward(self, x):
        x = self.conv1(x)
        xl = self.maxpool2_1(x)
        xr = self.conv2_1(x)
        x = torch.concat([xl, xr], dim=1)
        
        xl = self.conv3_l(x)
        xr = self.conv3_r(x)
        
        x = torch.concat([xl, xr], dim=1)
        
        xl = self.conv4_1(x)
        xr = self.maxpool4_1(x)
        return torch.concat([xl, xr], dim=1)


class InceptionA(nn.Module):
    def __init__(self):
        super(InceptionA ,self).__init__()
        
        self.convll = nn.Sequential(
            nn.AvgPool2d((3, 3), 1, padding=1),
            *(conv_block(384, 96, (1, 1), stride=1, padding=0))
        )
        
        self.convl = nn.Sequential(
            *(conv_block(384, 96, (1, 1), stride=1, padding=0))
        )
        
        self.convr = nn.Sequential(
            *(conv_block(384, 64, (1, 1), stride=1, padding=0) + 
              conv_block(64, 96, (3, 3), stride=1, padding=1))
        )
        
        self.convrr = nn.Sequential(
            *(conv_block(384, 64, (1, 1), stride=1, padding=0) + 
              conv_block(64, 96, (3, 3), stride=1, padding=1) +
              conv_block(96, 96, (3, 3), stride=1, padding=1))
        )
    
    def forward(self, x):
        xll = self.convll(x)
        xl = self.convl(x)
        xr = self.convr(x)
        xrr = self.convrr(x)
        
        x = torch.concat([xll, xl, xr, xrr], dim=1)
        return x


