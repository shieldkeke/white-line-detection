import torch.nn as nn
import torch
class MobileNet(nn.Module):# structure is modified by CYK 
    def __init__(self):
        super(MobileNet, self).__init__()

        def conv_bn(inp, oup, stride): #standard conv + batchnorm + relu
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride): #depthwise conv
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False), 
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
    
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False), 
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential( 		
            conv_bn(  3,  32, 2),   				
            conv_dw( 32,  64, 1),   				
            conv_dw( 64, 128, 2),   				
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            #conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 2048, 2),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(2048, 2)
        #self.fc = nn.Linear(2048, 512)
        #self.fc1 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 2048)
        x = self.fc(x)
        #x = self.fc1(x)
        return x

class hswish(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu6 = nn.ReLU6(inplace=True)
    def forward(self, x):
        out = x*self.relu6(x+3)/6
        return out
class hsigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu6 = nn.ReLU6(inplace=True)
    def forward(self, x):
        out = self.relu6(x+3)/6
        return out
class SE(nn.Module):
    def __init__(self, in_channels, reduce=4):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels//reduce, 1, bias=False),
            nn.BatchNorm2d(in_channels//reduce),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels // reduce, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            hsigmoid()
        )
    def forward(self, x):
        out = self.se(x)
        out = x*out
        return out
class Block(nn.Module):
    def __init__(self, kernel_size, in_channels, expand_size, out_channels, stride, se=False, nolinear='RE'):
        super().__init__()
        self.se = nn.Sequential()
        if se:
            self.se = SE(expand_size)
        if nolinear == 'RE':
            self.nolinear = nn.ReLU6(inplace=True)
        elif nolinear == 'HS':
            self.nolinear = hswish()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, expand_size, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(expand_size),
            self.nolinear,
            nn.Conv2d(expand_size, expand_size, kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=expand_size, bias=False),
            nn.BatchNorm2d(expand_size),
            self.se,
            self.nolinear,
            nn.Conv2d(expand_size, out_channels, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Sequential()
        if stride == 1 and in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.stride = stride
    def forward(self, x):
        out = self.block(x)
        if self.stride == 1:
            out += self.shortcut(x)
        return out
class MobileNetV3_Small(nn.Module):
    def __init__(self, output_num = 2):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            hswish()
        )
        self.neck = nn.Sequential(
            Block(3, 16, 16, 16, 2, se=True),
            Block(3, 16, 72, 24, 2),
            Block(3, 24, 88, 24, 1),
            Block(5, 24, 96, 40, 2, se=True, nolinear='HS'),
            Block(5, 40, 240, 40, 1, se=True, nolinear='HS'),
            Block(5, 40, 240, 40, 1, se=True, nolinear='HS'),
            Block(5, 40, 120, 48, 1, se=True, nolinear='HS'),
            Block(5, 48, 144, 48, 1, se=True, nolinear='HS'),
            Block(5, 48, 288, 96, 2, se=True, nolinear='HS'),
            Block(5, 96, 576, 96, 1, se=True, nolinear='HS'),
        
            Block(5, 96, 576, 96, 1, se=True, nolinear='HS'),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 576, 1, bias=False),
            nn.BatchNorm2d(576),
            hswish()
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv3 = nn.Sequential(
            nn.Conv2d(576, 1280, 1, bias=False),
            nn.BatchNorm2d(1280),
            hswish()
        )
        self.conv4 = nn.Conv2d(1280, output_num, 1, bias=False)
        self.fc = nn.Linear(600,2)
    def forward(self, x):
        x = self.conv1(x)
        x = self.neck(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

if __name__=='__main__':
    #net = MobileNet()
    net = MobileNetV3_Small()
    x = torch.randn(2,3,640,480)
    y = net(x)
    print(y.size())