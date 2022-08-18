import torch
import torch.nn as nn


class Conv1x1(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Conv1x1, self).__init__()
        self.conv1x1 = nn.Sequential(nn.Conv2d(input_channels, output_channels, 1, stride=1, padding=0, bias=False),
                                     nn.BatchNorm2d(output_channels),
                                     nn.ReLU(inplace=True)
                                     )

    def forward(self, x):
        return self.conv1x1(x)


class Head(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Head, self).__init__()
        self.conv5x5 = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, 5, 1, 2, groups=input_channels, bias=False),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(input_channels, output_channels, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(output_channels)
            )

    def forward(self, x):
        return self.conv5x5(x)


class Detect_FastestDet(nn.Module):
    stride = [8, 16, 32]

    def __init__(self, input_channels, category_num, anchors):
        super(Detect_FastestDet, self).__init__()
        self.nc = category_num          # number of classes
        self.no = category_num + 5      # number of outputs per anchor
        self.nl = len(anchors)          # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors

        self.conv1x1 = Conv1x1(input_channels, input_channels)

        self.obj_layers = Head(input_channels, 1)
        self.reg_layers = Head(input_channels, 4)
        self.cls_layers = Head(input_channels, self.nc)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1x1(x)

        obj = self.sigmoid(self.obj_layers(x))
        reg = self.reg_layers(x)
        cls = self.softmax(self.cls_layers(x))
        if self.training:
            return torch.cat((obj, reg, cls), dim=1)
        else:
            bs, _, H, W = reg.shape
            device = reg.device
            outputs = torch.cat((reg, obj, cls), dim=1)
            outputs = outputs.view(bs, self.no, -1).permute(0, 2, 1)
            gy, gx = torch.meshgrid([torch.arange(H), torch.arange(W)])
            gy = gy.reshape(1, H * W)
            gx = gx.reshape(1, H * W)
            bw, bh = outputs[..., 2].sigmoid(), outputs[..., 3].sigmoid()
            bcx = (outputs[..., 0].tanh() + gx.to(device)) / W
            bcy = (outputs[..., 1].tanh() + gy.to(device)) / H
            outputs[..., 0], outputs[..., 1] = bcx * W * 16, bcy * H * 16
            outputs[..., 2], outputs[..., 3] = bw * W * 16, bh * H * 16
            outputs[..., 4] = outputs[..., 4] ** 0.6
            outputs[..., 5:] = (outputs[..., 5:].max(dim=-1)[0] ** 0.4).unsqueeze(2)

            return (outputs, )
