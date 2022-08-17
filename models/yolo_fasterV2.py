"""YOLO-fasterV2 modules

Usage:
    $ python path/to/models/yolo.py --cfg yolo-fasterV2.yaml
"""

import sys
from pathlib import Path
import logging
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[1].as_posix())

from models.common import *
from utils.loss import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


try:
    import thop
except ImportError:
    thop = None

LOGGER = logging.getLogger(__name__)

class DetectFaster(nn.Module):
    onnx_dynamic = False  # ONNX export parameter
    export = False  # export mode
    def __init__(self, num_classes, anchors=(), in_channels=(72, 72, 72, 72), inplace=True, prior_prob=1e-2):
        super(DetectFaster, self).__init__()
        out_depth = 72
        self.num_classes = num_classes
        self.nc = self.num_classes
        self.no = self.nc + 5  # number of outputs per anchor
        self.nl = len(anchors) #number of detection layers
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        self.na = len(anchors[0]) // 2  # number of anchors
        self.output_reg_layers = nn.Conv2d(out_depth, 4 * self.na , 1, 1, 0)
        self.output_obj_layers = nn.Conv2d(out_depth, self.na , 1, 1, 0)
        self.output_cls_layers = nn.Conv2d(out_depth, self.nc, 1, 1, 0)
        self.inplace = inplace
        self.m = nn.ModuleList([self.output_reg_layers, self.output_obj_layers, self.output_cls_layers])
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))
        self.prior_prob = prior_prob
        self.layer_index = [0, 0, 0, 1, 1, 1]


    def initialize_biases(self):
        for _, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                if module.bias is not None:
                    init.constant_(module.bias, 0)
        
       
    def forward(self, xin):
        preds = [xin[0], xin[1], xin[1], xin[2], xin[3], xin[3]]
        for i in range(self.nl):
            preds[i * 3] = self.m[0](preds[i * 3])
            preds[(i * 3) + 1] = self.m[1](preds[(i * 3) + 1])
            preds[(i * 3) + 2] = self.m[2](preds[(i * 3) + 2])
        
        if self.training:
            return preds[0], preds[1], preds[2], preds[3], preds[4], preds[5]
 
        else:
            z = []
            for i in range(self.nl):
                bs, _, h, w = preds[i * 3].shape
                if self.export:
                    bs = -1
                reg_preds = preds[i * 3].view(bs, self.na, 4, h, w).sigmoid()
                obj_preds = preds[(i * 3) + 1].view(bs, self.na, 1, h, w).sigmoid()
                cls_preds = preds[(i * 3) + 2].view(bs, 1, self.nc, h, w).repeat(1, 3, 1, 1, 1)
                cls_preds = F.softmax(cls_preds, dim=2)
                x = torch.cat([reg_preds, obj_preds, cls_preds], 2)
                y = x.view(bs, self.na, self.no, h, w).permute(0, 1, 3, 4, 2).contiguous()
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x.shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(w, h, i)

                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2 + self.grid[i]) * self.stride[i * 3]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:
                    xy, wh, conf = y.split((2, 2, self.nc + 1), 4)  # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
                    xy = (xy * 2 + self.grid[i]) * self.stride[i * 3]  # xy
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, -1, self.no))
                output = torch.cat(z, 1)

            return (output,)


    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
            yv, xv = torch.meshgrid(y, x, indexing='ij')
        else:
            yv, xv = torch.meshgrid(y, x)
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i * 3]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid



