import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
import torch
import torch.nn.functional as F

__all__ = ['ResNet_dcd', 'resnet18_dcd', 'resnet34_dcd', 'resnet50_dcd', 'resnet101_dcd', 'resnet152_dcd']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 3.

class SEModule_small(nn.Module):
    def __init__(self, channel):
        super(SEModule_small, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel, bias=False),
            Hsigmoid()
        )

    def forward(self, x):
        y = self.fc(x)
        return x * y

class DYCls(nn.Module):
    def __init__(self, inp, oup):
        super(DYCls, self).__init__()
        self.dim = 32
        self.cls = nn.Linear(inp, oup)
        self.cls_q = nn.Linear(inp, self.dim, bias=False)
        self.cls_p = nn.Linear(self.dim, oup, bias=False)

        mid = 32

        self.fc = nn.Sequential(
            nn.Linear(inp, mid, bias=False),
            SEModule_small(mid),
        )
        self.fc_phi = nn.Linear(mid, self.dim**2, bias=False)
        self.fc_scale = nn.Linear(mid, oup, bias=False)
        self.hs = Hsigmoid()
        self.bn1 = nn.BatchNorm1d(self.dim)
        self.bn2 = nn.BatchNorm1d(self.dim)

    def forward(self, x):
        #r = self.cls(x)
        b, c = x.size()
        y = self.fc(x)
        dy_phi = self.fc_phi(y).view(b, self.dim, self.dim)
        dy_scale = self.hs(self.fc_scale(y)).view(b, -1)

        r = dy_scale*self.cls(x)

        x = self.cls_q(x)
        x = self.bn1(x)
        x = self.bn2(torch.matmul(dy_phi, x.view(b, self.dim, 1)).view(b, self.dim)) + x
        x = self.cls_p(x)

        return x + r

class conv_basic_dy(nn.Module):
    def __init__(self, inplanes, planes, stride):
        super(conv_basic_dy, self).__init__()
        
        self.conv = conv3x3(inplanes, planes, stride)
        self.dim = int(math.sqrt(inplanes*4))
        squeeze = max(inplanes*4, self.dim ** 2) // 16
        if squeeze < 4:
            squeeze = 4
        
        self.q = nn.Conv2d(inplanes, self.dim, 1, stride, 0, bias=False)

        self.p = nn.Conv2d(self.dim, planes, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(self.dim)
        self.bn2 = nn.BatchNorm1d(self.dim)

        self.avg_pool = nn.AdaptiveAvgPool2d(2)  

        self.fc = nn.Sequential(
            nn.Linear(inplanes*4, squeeze, bias=False),
            SEModule_small(squeeze),
        ) 
        self.fc_phi = nn.Linear(squeeze, self.dim**2, bias=False)
        self.fc_scale = nn.Linear(squeeze, planes, bias=False)
        self.hs = Hsigmoid()     
        
    def forward(self, x):
        r = self.conv(x)
        b, c, _, _= x.size()
        y = self.avg_pool(x).view(b, c*4)
        y = self.fc(y)
        phi = self.fc_phi(y).view(b, self.dim, self.dim)
        scale = self.hs(self.fc_scale(y)).view(b,-1,1,1)
        r = scale.expand_as(r)*r        

        out = self.bn1(self.q(x))
        _, _, h, w = out.size()

        out = out.view(b,self.dim,-1)
        out = self.bn2(torch.matmul(phi, out)) + out
        out = out.view(b,-1,h,w)        
        out = self.p(out) + r
         
        return out    

class BasicBlock_dy(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock_dy, self).__init__()

        self.inplanes = inplanes
        self.planes = planes       
        self.conv1 = conv_basic_dy(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv_basic_dy(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        
        residual = x
        b, c, _, _= x.size()
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class conv_dy(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, stride, padding):
        super(conv_dy, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.dim = int(math.sqrt(inplanes))
        squeeze = max(inplanes, self.dim ** 2) // 16

        self.q = nn.Conv2d(inplanes, self.dim, 1, stride, 0, bias=False)

        self.p = nn.Conv2d(self.dim, planes, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(self.dim)
        self.bn2 = nn.BatchNorm1d(self.dim)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)  

        self.fc = nn.Sequential(
            nn.Linear(inplanes, squeeze, bias=False),
            SEModule_small(squeeze),
        )
        self.fc_phi = nn.Linear(squeeze, self.dim**2, bias=False)
        self.fc_scale = nn.Linear(squeeze, planes, bias=False)
        self.hs = Hsigmoid()  
 
    def forward(self, x):
        r = self.conv(x)
        b, c, _, _= x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y)
        phi = self.fc_phi(y).view(b, self.dim, self.dim)
        scale = self.hs(self.fc_scale(y)).view(b,-1,1,1)
        r = scale.expand_as(r)*r

        out = self.bn1(self.q(x))
        _, _, h, w = out.size()

        out = out.view(b,self.dim,-1)
        out = self.bn2(torch.matmul(phi, out)) + out
        out = out.view(b,-1,h,w)        
        out = self.p(out) + r
        return out

class Bottleneck_dy(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck_dy, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.conv1 = conv_dy(inplanes, planes, 1, 1, 0)
        self.bn1 = nn.BatchNorm2d(planes)        
        self.conv2 = conv_dy(planes, planes, 3, stride, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv_dy(planes, planes * 4, 1, 1, 0)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet_dcd(nn.Module):

    def __init__(self, block, layers, num_classes=1000, dropout=None):
        self.inplanes = 64
        super(ResNet_dcd, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        classifier = []
        if dropout is not None:
            classifier.append(nn.Dropout(p=dropout))
        classifier.append(DYCls(512 * block.expansion, num_classes))
        #classifier.append(nn.Linear(512 * block.expansion, num_classes))
        self.classifier = nn.Sequential(*classifier)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x


def resnet18_dcd(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_dcd(BasicBlock_dy, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34_dcd(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_dcd(BasicBlock_dy, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50_dcd(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_dcd(Bottleneck_dy, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101_dcd(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_dcd(Bottleneck_dy, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152_dcd(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_dcd(Bottleneck_dy, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
