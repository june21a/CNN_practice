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



class ReductionA(nn.Module):
    def __init__(self):
        super(ReductionA, self).__init__()
        
        self.convl = nn.MaxPool2d((3, 3), stride=2, padding=0)
        self.convc = nn.Sequential(*(conv_block(384, 384, (3, 3), stride=2, padding=0)))
        self.convr = nn.Sequential(*(
            conv_block(384, 192, (1, 1), stride=1, padding=0) +
            conv_block(192, 224, (3, 3), stride=1, padding=1) +
            conv_block(224, 256, (3, 3), stride=2, padding=0)
        )) 
    
    def forward(self, x):
        xl = self.convl(x)
        xc = self.convc(x)
        xr = self.convr(x)
        x = torch.concat([xl, xc, xr], dim=1)
        return x



class InceptionB(nn.Module):
    def __init__(self):
        super(InceptionB, self).__init__()
        
        self.convll = nn.Sequential(
            nn.AvgPool2d((3, 3), 1, padding=1),
            *(conv_block(1024, 128, (1, 1), stride=1, padding=0))
        )
        
        self.convl = nn.Sequential(*(
            conv_block(1024, 384, (1, 1), stride=1, padding=0)
        ))
        
        self.convr = nn.Sequential(*(
            conv_block(1024, 192, (1, 1), stride=1, padding=0) +
            conv_block(192, 224, (1, 7), stride=1, padding=(0, 3)) +
            conv_block(224, 256, (7, 1), stride=1, padding=(3, 0))
        ))
        
        self.convrr = nn.Sequential(*(
            conv_block(1024, 192, (1, 1), stride=1, padding=0) +
            conv_block(192, 192, (1, 7), stride=1, padding=(0, 3)) +
            conv_block(192, 224, (7, 1), stride=1, padding=(3, 0)) +
            conv_block(224, 224, (1, 7), stride=1, padding=(0, 3)) +
            conv_block(224, 256, (7, 1), stride=1, padding=(3, 0))
        ))
    
    
    def forward(self, x):
        xll = self.convll(x)
        xl = self.convl(x)
        xr = self.convr(x)
        xrr = self.convrr(x)
        
        x = torch.concat([xll, xl, xr, xrr], dim=1)
        return x


class InceptionC(nn.Module):
    def __init__(self):
        super(InceptionC, self).__init__()

        self.convll = nn.Sequential(
            nn.AvgPool2d((3, 3), 1, padding=1),
            *(conv_block(1536, 256, (1, 1), stride=1, padding=0))
        )
        
        self.convl = nn.Sequential(
            *(conv_block(1536, 256, (1, 1), stride=1, padding=0))
        )
        
        self.convr_bottlenecek = nn.Sequential(*(
            conv_block(1536, 384, (1, 1), stride=1, padding=0)
        ))
        self.convr_l = nn.Sequential(*(
            conv_block(384, 256, (1, 3), stride=1, padding=(0, 1))
        ))
        self.convr_r = nn.Sequential(*(
            conv_block(384, 256, (3, 1), stride=1, padding=(1, 0))
        ))
        
        
        self.convrr_bottleneck = nn.Sequential(*(
            conv_block(1536, 384, (1, 1), stride=1, padding=0) +
            conv_block(384, 448, (1, 3), stride=1, padding=(0, 1)) +
            conv_block(448, 512, (3, 1), stride=1, padding=(1, 0))
        ))
        self.convrr_l = nn.Sequential(*(
            conv_block(512, 256, (3, 1), stride=1, padding=(1, 0))
        ))
        self.convrr_r = nn.Sequential(*(
            conv_block(512, 256, (1, 3), stride=1, padding=(0, 1))
        ))
    
    
    def forward(self, x):
        xll = self.convll(x)
        xl = self.convl(x)
        
        xr = self.convr_bottlenecek(x)
        xr_l = self.convr_l(xr)
        xr_r = self.convr_r(xr)
        
        xrr = self.convrr_bottleneck(x)
        xrr_l = self.convrr_l(xrr)
        xrr_r = self.convrr_r(xrr)
        
        x = torch.concat([xll, xl, xr_l, xr_r, xrr_l, xrr_r], dim=1)
        return x

    