import torch.nn as nn
import torch.nn.functional as F
import torch

class Quant_Resnet50(nn.Module):
    def __init__(self):
        super(Quant_Resnet50, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3)
        self.mxpl = nn.MaxPool2d(3, 2, 1)
        self.stage1 = nn.Sequential(
            Bottleneck1(64, 256, False),
            Bottleneck2(256),
            Bottleneck2(256),
        )
        self.stage2 = nn.Sequential(
            Bottleneck1(256, 512, True),
            Bottleneck2(512),
            Bottleneck2(512),
            Bottleneck2(512)
        )
        self.stage3 = nn.Sequential(
            Bottleneck1(512, 1024, True),
            Bottleneck2(1024),
            Bottleneck2(1024),
            Bottleneck2(1024),
            Bottleneck2(1024),
            Bottleneck2(1024)

        )
        self.stage4 = nn.Sequential(
            Bottleneck1(1024, 2048, True),
            Bottleneck2(2048),
            Bottleneck2(2048),
        )
        self.avpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(2048, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.mxpl(out)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.avpool(out)
        out = out.reshape(x.shape[0],-1)
        out = self.fc(out)
        return out

class Bottleneck1(nn.Module):
    def __init__(self,in_channel, out_channel, down_sample):
        super(Bottleneck1, self).__init__()
        if down_sample == True:
            stride = 2
        else:
            stride = 1

        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, stride=stride),
            nn.BatchNorm2d(out_channel),
        )

        mid_channel = out_channel // 4
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, 1, stride=stride, padding=0),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(),
            nn.Conv2d(mid_channel, mid_channel, 3, 1, 1),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(),
            nn.Conv2d(mid_channel, out_channel, 1, 1, 0),
            nn.BatchNorm2d(out_channel),
        )

    def forward(self, x):
        x = self.quant(x)
        out = self.conv2(x)
        out = F.relu(out)
        out = self.dequant(out)
        return out


class Bottleneck2(nn.Module):
    def __init__(self,out_channel):
        super(Bottleneck2,self).__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()
        mid_channel = out_channel // 4

        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channel, mid_channel, 1, 1, 0),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(),
            nn.Conv2d(mid_channel, mid_channel, 3, 1, 1),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(),
            nn.Conv2d(mid_channel, out_channel, 1, 1, 0),
            nn.BatchNorm2d(out_channel)
        )

    def forward(self, x):
        x = self.quant(x)
        out = self.conv3(x)
        out = F.relu(out + x)
        out = self.dequant(out)
        return out