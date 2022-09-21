import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils.anchor_generator import generate_anchors
from utils.loss import dist2bbox


class ConvBNLayer(nn.Module):

    def __init__(self,
                 ch_in,
                 ch_out,
                 filter_size=3,
                 stride=1,
                 groups=1,
                 padding=0,
                 act=None):
        super(ConvBNLayer, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=ch_in,
            out_channels=ch_out,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False
        )
        self.bn = nn.BatchNorm2d(ch_out, )
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)

        return x


def get_activation(name="silu", inplace=True):
    if name is None:
        return nn.Identity()

    if isinstance(name, str):
        if name == "silu":
            module = nn.SiLU(inplace=inplace)
        elif name == "relu":
            module = nn.ReLU(inplace=inplace)
        elif name == "lrelu":
            module = nn.LeakyReLU(0.1, inplace=inplace)
        elif name == 'swish':
            module = Swish(inplace=inplace)
        elif name == 'hardsigmoid':
            module = nn.Hardsigmoid(inplace=inplace)
        else:
            raise AttributeError("Unsupported act type: {}".format(name))
        return module
    elif isinstance(name, nn.Module):
        return name
    else:
        raise AttributeError("Unsupported act type: {}".format(name))


class Swish(nn.Module):
    def __init__(self, inplace=True):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x.mul_(F.sigmoid(x))
            return x
        else:
            return x * F.sigmoid(x)


class ESEAttn(nn.Module):
    def __init__(self, feat_channels, act='swish'):
        super(ESEAttn, self).__init__()
        self.fc = nn.Conv2d(feat_channels, feat_channels, 1)
        self.sig = nn.Sigmoid()
        self.conv = ConvBNLayer(feat_channels, feat_channels, 1, act=act)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.fc.weight, mean=0, std=0.001)

    def forward(self, feat):
        avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
        weight = self.sig(self.fc(avg_feat))
        return self.conv(feat * weight)


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


class DetectE(nn.Module):
    '''Efficient Decoupled Head
    With hardware-aware degisn, the decoupled head is optimized with
    hybridchannels methods.
    '''

    def __init__(self,
                 num_classes=80,
                 channels_list=[64, 128, 256],
                 anchors=1,
                 num_layers=3,
                 inplace=True,
                 use_dfl=True,
                 reg_max=16):  # detection layer
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
        self.use_dfl = use_dfl
        self.reg_max = reg_max
        self.proj_conv = nn.Conv2d(self.reg_max + 1, 1, 1, bias=False)
        self.grid_cell_offset = 0.5
        self.grid_cell_size = 5.0

        # Init decouple head
        self.stems = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.head_layers = build_effidehead_layer(channels_list, 1, self.nc)

        # Efficient decoupled head layers
        for i in range(num_layers):
            idx = i * 5
            self.stems.append(self.head_layers[idx])
            self.cls_convs.append(self.head_layers[idx + 1])
            self.reg_convs.append(self.head_layers[idx + 2])
            self.cls_preds.append(self.head_layers[idx + 3])
            self.reg_preds.append(self.head_layers[idx + 4])

    def initialize_biases(self):
        for conv in self.cls_preds:
            b = conv.bias.view(-1, )
            b.data.fill_(-math.log((1 - self.prior_prob) / self.prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            w = conv.weight
            w.data.fill_(0.)
            conv.weight = torch.nn.Parameter(w, requires_grad=True)

        for conv in self.reg_preds:
            b = conv.bias.view(-1, )
            b.data.fill_(1.0)
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            w = conv.weight
            w.data.fill_(0.)
            conv.weight = torch.nn.Parameter(w, requires_grad=True)

        self.proj = nn.Parameter(torch.linspace(0, self.reg_max, self.reg_max + 1), requires_grad=False)
        self.proj_conv.weight = nn.Parameter(self.proj.view([1, self.reg_max + 1, 1, 1]).clone().detach(), requires_grad=False)

    def forward(self, x):
        if self.training:
            cls_score_list = []
            reg_distri_list = []

            for i in range(self.nl):
                x[i] = self.stems[i](x[i])
                cls_x = x[i]
                reg_x = x[i]
                cls_feat = self.cls_convs[i](cls_x)
                cls_output = self.cls_preds[i](cls_feat)
                reg_feat = self.reg_convs[i](reg_x)
                reg_output = self.reg_preds[i](reg_feat)

                cls_output = torch.sigmoid(cls_output)
                cls_score_list.append(cls_output.flatten(2).permute((0, 2, 1)))
                reg_distri_list.append(reg_output.flatten(2).permute((0, 2, 1)))

            cls_score_list = torch.cat(cls_score_list, axis=1)
            reg_distri_list = torch.cat(reg_distri_list, axis=1)

            return x, cls_score_list, reg_distri_list
        else:
            cls_score_list = []
            reg_dist_list = []
            anchor_points, stride_tensor = generate_anchors(x,
                                                            self.stride,
                                                            self.grid_cell_size,
                                                            self.grid_cell_offset,
                                                            device=x[0].device,
                                                            is_eval=True)

            for i in range(self.nl):
                b, _, h, w = x[i].shape
                l = h * w
                x[i] = self.stems[i](x[i])
                cls_x = x[i]
                reg_x = x[i]
                cls_feat = self.cls_convs[i](cls_x)
                cls_output = self.cls_preds[i](cls_feat)
                reg_feat = self.reg_convs[i](reg_x)
                reg_output = self.reg_preds[i](reg_feat)

                if self.use_dfl:
                    reg_output = reg_output.reshape([-1, 4, self.reg_max + 1, l]).permute(0, 2, 1, 3)
                    reg_output = self.proj_conv(F.softmax(reg_output, dim=1))

                cls_output = torch.sigmoid(cls_output)
                cls_score_list.append(cls_output.reshape([b, self.nc, l]))
                reg_dist_list.append(reg_output.reshape([b, 4, l]))

            cls_score_list = torch.cat(cls_score_list, axis=-1).permute(0, 2, 1)
            reg_dist_list = torch.cat(reg_dist_list, axis=-1).permute(0, 2, 1)

            pred_bboxes = dist2bbox(reg_dist_list, anchor_points, box_format='xywh')
            pred_bboxes *= stride_tensor
            return torch.cat(
                [
                    pred_bboxes,
                    torch.ones((b, pred_bboxes.shape[1], 1), device=pred_bboxes.device, dtype=pred_bboxes.dtype),
                    cls_score_list
                ],
                axis=-1)


def build_effidehead_layer(channels_list, num_anchors, num_classes, reg_max=16):
    head_layers = nn.Sequential(
        # stem0
        ESEAttn(feat_channels=channels_list[6], act='swish'),
        # cls_conv0
        nn.Conv2d(in_channels=channels_list[6], out_channels=channels_list[6], kernel_size=3, stride=1),
        # reg_conv0
        nn.Conv2d(in_channels=channels_list[6], out_channels=channels_list[6], kernel_size=3, stride=1),
        # cls_pred0
        nn.Conv2d(in_channels=channels_list[6], out_channels=num_classes * num_anchors, kernel_size=1),
        # reg_pred0
        nn.Conv2d(in_channels=channels_list[6], out_channels=4 * (reg_max + num_anchors), kernel_size=1),

        # stem1
        ESEAttn(feat_channels=channels_list[8], act='swish'),
        # cls_conv1
        nn.Conv2d(in_channels=channels_list[8], out_channels=channels_list[8], kernel_size=3, stride=1),
        # reg_conv1
        nn.Conv2d(in_channels=channels_list[8], out_channels=channels_list[8], kernel_size=3, stride=1),
        # cls_pred1
        nn.Conv2d(in_channels=channels_list[8], out_channels=num_classes * num_anchors, kernel_size=1),
        # reg_pred1
        nn.Conv2d(in_channels=channels_list[8], out_channels=4 * (reg_max + num_anchors), kernel_size=1),

        # stem2
        ESEAttn(feat_channels=channels_list[10], act='swish'),
        # cls_conv2
        nn.Conv2d(in_channels=channels_list[10], out_channels=channels_list[10], kernel_size=3, stride=1),
        # reg_conv2
        nn.Conv2d(in_channels=channels_list[10], out_channels=channels_list[10], kernel_size=3, stride=1),
        # cls_pred2
        nn.Conv2d(in_channels=channels_list[10], out_channels=num_classes * num_anchors, kernel_size=1),
        # reg_pred2
        nn.Conv2d(in_channels=channels_list[10], out_channels=4 * (reg_max + num_anchors), kernel_size=1)
    )
    return head_layers