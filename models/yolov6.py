import torch
import torch.nn as nn
import math


class Conv(nn.Module):
    '''Normal Conv with SiLU activation'''
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1, bias=False):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class Detectv6(nn.Module):
    '''
    Efficient Decoupled Head
    With hardware-aware degisn, the decoupled head is optimized with hybridchannels methods.
    '''
    def __init__(self,
                 num_classes=80,
                 channels_list=[64, 128, 256],
                 anchors=1,
                 num_layers=3,
                 inplace=True):  # detection layer
        super().__init__()
        self.nc = num_classes  # number of classes
        self.no = num_classes + 5  # number of outputs per anchor
        self.nl = num_layers  # number of detection layers
        if isinstance(anchors, (list, tuple)):
            self.na = len(anchors[0]) // 2
        else:
            self.na = anchors
        self.anchors = anchors
        self.grid = [torch.zeros(1)] * num_layers
        self.prior_prob = 1e-2
        self.inplace = inplace
        stride = [8, 16, 32]  # strides computed during build
        self.stride = torch.tensor(stride)
        self.head_layers = build_effidehead_layer(channels_list, 1, self.nc)

        # Init decouple head
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()

        # Efficient decoupled head layers
        for i in range(num_layers):
            idx = i*6
            self.stems.append(self.head_layers[idx])
            self.cls_convs.append(self.head_layers[idx+1])
            self.reg_convs.append(self.head_layers[idx+2])
            self.cls_preds.append(self.head_layers[idx+3])
            self.reg_preds.append(self.head_layers[idx+4])
            self.obj_preds.append(self.head_layers[idx+5])

    def initialize_biases(self):
        for conv in self.cls_preds:
            b = conv.bias.view(self.na, -1)
            b.data.fill_(-math.log((1 - self.prior_prob) / self.prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        for conv in self.obj_preds:
            b = conv.bias.view(self.na, -1)
            b.data.fill_(-math.log((1 - self.prior_prob) / self.prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, x):

        z = []
        for i in range(self.nl):
            x[i] = self.stems[i](x[i])
            cls_x = x[i]
            reg_x = x[i]
            cls_feat = self.cls_convs[i](cls_x)
            cls_output = self.cls_preds[i](cls_feat)
            reg_feat = self.reg_convs[i](reg_x)
            reg_output = self.reg_preds[i](reg_feat)
            obj_output = self.obj_preds[i](reg_feat)

            if self.training:
                x[i] = torch.cat([reg_output, obj_output, cls_output], 1)
                bs, _, ny, nx = x[i].shape
                x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            else:
                y = torch.cat([reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1)
                bs, _, ny, nx = y.shape
                y = y.view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

                if self.grid[i].shape[2:4] != y.shape[2:4]:
                    d = self.stride.device
                    yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)])
                    self.grid[i] = torch.stack((xv, yv), 2).view(1, self.na, ny, nx, 2).float()

                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] + self.grid[i].to('cuda:0')) * self.stride[i].to('cuda:0')  # xy
                    y[..., 2:4] = torch.exp(y[..., 2:4]) * self.stride[i].to('cuda:0')   # wh
                else:
                    xy = (y[..., 0:2] + self.grid[i]) * self.stride[i]  # xy
                    wh = torch.exp(y[..., 2:4]) * self.stride[i]  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1),)


def build_effidehead_layer(channels_list, num_anchors, num_classes):
    head_layers = nn.Sequential(
        # stem0
        Conv(in_channels=channels_list[0], out_channels=channels_list[0], kernel_size=1, stride=1),
        # cls_conv0
        Conv(in_channels=channels_list[0], out_channels=channels_list[0], kernel_size=3, stride=1),
        # reg_conv0
        Conv(in_channels=channels_list[0], out_channels=channels_list[0], kernel_size=3, stride=1),
        # cls_pred0
        nn.Conv2d(in_channels=channels_list[0], out_channels=num_classes * num_anchors, kernel_size=1),
        # reg_pred0
        nn.Conv2d(in_channels=channels_list[0], out_channels=4 * num_anchors, kernel_size=1),
        # obj_pred0
        nn.Conv2d(in_channels=channels_list[0], out_channels=1 * num_anchors, kernel_size=1),
        # stem1
        Conv(in_channels=channels_list[1], out_channels=channels_list[1], kernel_size=1, stride=1),
        # cls_conv1
        Conv(in_channels=channels_list[1], out_channels=channels_list[1], kernel_size=3, stride=1),
        # reg_conv1
        Conv(in_channels=channels_list[1], out_channels=channels_list[1], kernel_size=3, stride=1),
        # cls_pred1
        nn.Conv2d(in_channels=channels_list[1], out_channels=num_classes * num_anchors, kernel_size=1),
        # reg_pred1
        nn.Conv2d(in_channels=channels_list[1], out_channels=4 * num_anchors, kernel_size=1),
        # obj_pred1
        nn.Conv2d(in_channels=channels_list[1], out_channels=1 * num_anchors, kernel_size=1),
        # stem2
        Conv(in_channels=channels_list[2], out_channels=channels_list[2], kernel_size=1, stride=1),
        # cls_conv2
        Conv(in_channels=channels_list[2], out_channels=channels_list[2], kernel_size=3, stride=1),
        # reg_conv2
        Conv(in_channels=channels_list[2], out_channels=channels_list[2], kernel_size=3, stride=1),
        # cls_pred2
        nn.Conv2d(in_channels=channels_list[2], out_channels=num_classes * num_anchors, kernel_size=1),
        # reg_pred2
        nn.Conv2d(in_channels=channels_list[2], out_channels=4 * num_anchors, kernel_size=1),
        # obj_pred2
        nn.Conv2d(in_channels=channels_list[2], out_channels=1 * num_anchors, kernel_size=1)
    )
    return head_layers
