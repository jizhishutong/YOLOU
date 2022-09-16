import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from models.experimental import *
from utils.general import check_version, check_yaml, print_args
try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None


class Detect_YOLOSA(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.is_dh = True
        self.is_ndh = True
        if self.is_dh:
            if self.is_ndh:
                self.m = nn.ModuleList(Conv(x, x // 2) for x in ch)
                self.m2 = nn.ModuleList(
                    nn.Sequential(*[Conv(x // 2, x // 2, k=3)]) for x in ch)  # , Conv(x//2, x//2, k=3)
                self.m2_cls = nn.ModuleList(nn.Conv2d(x // 2, self.na * self.nc, 1, padding=0) for x in ch)
                self.m2_obj = nn.ModuleList(nn.Conv2d(x // 2, self.na * 1, 1, padding=0) for x in ch)
                self.m2_reg = nn.ModuleList(nn.Conv2d(x // 2, self.na * 4, 1, padding=0) for x in ch)
            else:
                self.m = nn.ModuleList(Conv(x, 256) for x in ch)
                self.m2 = nn.ModuleList(
                    nn.Sequential(*[Conv(256, 256, k=3), Conv(256, 256, k=3)]) for x in ch)  # , Conv(x//2, x//2, k=3)
                self.m2_cls = nn.ModuleList(nn.Conv2d(256, self.na * self.nc, 1, padding=0) for x in ch)
                self.m2_obj = nn.ModuleList(nn.Conv2d(256, self.na * 1, 1, padding=0) for x in ch)
                self.m2_reg = nn.ModuleList(nn.Conv2d(256, self.na * 4, 1, padding=0) for x in ch)
        else:
            self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            if self.is_dh:
                if self.is_ndh:
                    x1 = self.m[i](x[i])
                    x11 = self.m2[i](x1)
                    x21 = self.m2[i](x1)
                    # x(bs,240,20,20)
                    x11_cls = self.m2_cls[i](x11)
                    # x(bs,3,20,20)
                    x21_obj = self.m2_obj[i](x21)
                    # x(bs,12,20,20)
                    x21_reg = self.m2_reg[i](x21)

                    bs, _, ny, nx = x11_cls.shape
                    x11_cls = x11_cls.view(bs, self.na, self.nc, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
                    x21_obj = x21_obj.view(bs, self.na, 1, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
                    x21_reg = x21_reg.view(bs, self.na, 4, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

                    x[i] = torch.cat([x21_reg, x21_obj, x11_cls], -1)
                else:
                    x1 = self.m[i](x[i])
                    x11 = self.m2[i](x1)
                    x21 = self.m2[i](x1)
                    # x(bs,240,20,20)
                    x11_cls = self.m2_cls[i](x11)
                    # x(bs,3,20,20)
                    x21_obj = self.m2_obj[i](x21)
                    # x(bs,12,20,20)
                    x21_reg = self.m2_reg[i](x21)

                    bs, _, ny, nx = x11_cls.shape
                    x11_cls = x11_cls.view(bs, self.na, self.nc, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
                    x21_obj = x21_obj.view(bs, self.na, 1, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
                    x21_reg = x21_reg.view(bs, self.na, 4, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

                    x[i] = torch.cat([x21_reg, x21_obj, x11_cls], -1)
            else:
                x[i] = self.m[i](x[i])  # conv
                bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
                x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
            yv, xv = torch.meshgrid([torch.arange(ny, device=d), torch.arange(nx, device=d)], indexing='ij')
        else:
            yv, xv = torch.meshgrid([torch.arange(ny, device=d), torch.arange(nx, device=d)])
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        anchor_grid = (self.anchors[i].clone() * self.stride[i]).view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()
        return grid, anchor_grid
