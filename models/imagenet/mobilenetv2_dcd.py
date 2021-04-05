"""
Creates a MobileNetV2 Model as defined in:
Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen. (2018). 
MobileNetV2: Inverted Residuals and Linear Bottlenecks
arXiv preprint arXiv:1801.04381.
import from https://github.com/tonylins/pytorch-mobilenet-v2
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = ['mobilenetv2_dcd']

class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 3.

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

class DYModule(nn.Module):
    def __init__(self, inp, oup, fc_squeeze=8):
        super(DYModule, self).__init__()
        self.conv = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)
        if inp < oup:
            self.mul = 4
            reduction = 8
            self.avg_pool = nn.AdaptiveAvgPool2d(2)
        else:
            self.mul = 1
            reduction=2
            self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.dim = min((inp*self.mul)//reduction, oup//reduction) 
        while self.dim ** 2 > inp * self.mul * 2:
            reduction *= 2
            self.dim = min((inp*self.mul)//reduction, oup//reduction) 
        if self.dim < 4:
            self.dim = 4

        squeeze = max(inp*self.mul, self.dim ** 2) // fc_squeeze
        if squeeze < 4:
            squeeze = 4
        self.conv_q = nn.Conv2d(inp, self.dim, 1, 1, 0, bias=False)
          
        self.fc = nn.Sequential(
            nn.Linear(inp*self.mul, squeeze, bias=False),
            SEModule_small(squeeze),
        )
        self.fc_phi = nn.Linear(squeeze, self.dim**2, bias=False)
        self.fc_scale = nn.Linear(squeeze, oup, bias=False)
        self.hs = Hsigmoid()
        self.conv_p = nn.Conv2d(self.dim, oup, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(self.dim)
        self.bn2 = nn.BatchNorm1d(self.dim)        

    def forward(self, x):
        r = self.conv(x)

        b, c, h, w = x.size()
        y = self.avg_pool(x).view(b, c*self.mul)
        y = self.fc(y)
        dy_phi = self.fc_phi(y).view(b, self.dim, self.dim)
        dy_scale = self.hs(self.fc_scale(y)).view(b,-1,1,1)
        r = dy_scale.expand_as(r)*r

        x = self.conv_q(x)
        x = self.bn1(x)
        x = x.view(b,-1,h*w)
        x = self.bn2(torch.matmul(dy_phi, x)) + x
        x = x.view(b,-1,h,w)
        x = self.conv_p(x)
        return x + r

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

class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            Hsigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, dy=False, fc_squeeze=8):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            if not dy:
                self.conv = nn.Sequential(
                    # pw
                    nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True),
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                )
            else:
                self.conv = nn.Sequential(
                    # pw
                    DYModule(inp, hidden_dim, fc_squeeze=fc_squeeze),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True),
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True),
                    # pw-linear
                    DYModule(hidden_dim, oup, fc_squeeze=fc_squeeze),
                    nn.BatchNorm2d(oup),                     
                )
                
    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2_dcd(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1., dropout=None, fc_squeeze=8):
        super(MobileNetV2_dcd, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s, dynamic
            [1,  16, 1, 1, True],
            [6,  24, 2, 2, True],
            [6,  32, 3, 2, True],
            [6,  64, 4, 2, True],
            [6,  96, 3, 1, True],
            [6, 160, 3, 2, True],
            [6, 320, 1, 1, True],
        ]

        # building first layer
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual
        for t, c, n, s, dy in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t, dy, fc_squeeze))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        output_channel = _make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1280
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        classifier = []
        if dropout is not None:
            classifier.append(nn.Dropout(p=dropout))
        classifier.append(DYCls(output_channel, num_classes))
        #classifier.append(nn.Linear(output_channel, num_classes))
        self.classifier = nn.Sequential(*classifier)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

def mobilenetv2_dcd(**kwargs):
    """
    Constructs a MobileNet V2 model
    """
    return MobileNetV2_dcd(**kwargs)

