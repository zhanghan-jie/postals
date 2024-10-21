# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""Experimental modules."""

import math

import numpy as np
import torch
import torch.nn as nn

from utils.downloads import attempt_download


class Sum(nn.Module):
    """Weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070."""

    def __init__(self, n, weight=False):
        """Initializes a module to sum outputs of layers with number of inputs `n` and optional weighting, supporting 2+
        inputs.
        """
        super().__init__()
        self.weight = weight  # apply weights boolean
        self.iter = range(n - 1)  # iter object
        if weight:
            self.w = nn.Parameter(-torch.arange(1.0, n) / 2, requires_grad=True)  # layer weights

    def forward(self, x):
        """Processes input through a customizable weighted sum of `n` inputs, optionally applying learned weights."""
        y = x[0]  # no weight
        if self.weight:
            w = torch.sigmoid(self.w) * 2
            for i in self.iter:
                y = y + x[i + 1] * w[i]
        else:
            for i in self.iter:
                y = y + x[i + 1]
        return y


class MixConv2d(nn.Module):
    """Mixed Depth-wise Conv https://arxiv.org/abs/1907.09595."""

    def __init__(self, c1, c2, k=(1, 3), s=1, equal_ch=True):
        """Initializes MixConv2d with mixed depth-wise convolutional layers, taking input and output channels (c1, c2),
        kernel sizes (k), stride (s), and channel distribution strategy (equal_ch).
        """
        super().__init__()
        n = len(k)  # number of convolutions
        if equal_ch:  # equal c_ per group
            i = torch.linspace(0, n - 1e-6, c2).floor()  # c2 indices
            c_ = [(i == g).sum() for g in range(n)]  # intermediate channels
        else:  # equal weight.numel() per group
            b = [c2] + [0] * n
            a = np.eye(n + 1, n, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            c_ = np.linalg.lstsq(a, b, rcond=None)[0].round()  # solve for equal weight indices, ax = b

        self.m = nn.ModuleList(
            [nn.Conv2d(c1, int(c_), k, s, k // 2, groups=math.gcd(c1, int(c_)), bias=False) for k, c_ in zip(k, c_)]
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        """Performs forward pass by applying SiLU activation on batch-normalized concatenated convolutional layer
        outputs.
        """
        return self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))


class Ensemble(nn.ModuleList):
    """Ensemble of models."""

    def __init__(self):
        """Initializes an ensemble of models to be used for aggregated predictions."""
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        """Performs forward pass aggregating outputs from an ensemble of models.."""
        y = [module(x, augment, profile, visualize)[0] for module in self]
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output


def attempt_load(weights, device=None, inplace=True, fuse=True):
    """
    Loads and fuses an ensemble or single YOLOv5 model from weights, handling device placement and model adjustments.

    Example inputs: weights=[a,b,c] or a single model weights=[a] or weights=a.
    """
    from models.yolo import Detect, Model

    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt = torch.load(attempt_download(w), map_location="cpu")  # load
        ckpt = (ckpt.get("ema") or ckpt["model"]).to(device).float()  # FP32 model

        # Model compatibility updates
        if not hasattr(ckpt, "stride"):
            ckpt.stride = torch.tensor([32.0])
        if hasattr(ckpt, "names") and isinstance(ckpt.names, (list, tuple)):
            ckpt.names = dict(enumerate(ckpt.names))  # convert to dict

        model.append(ckpt.fuse().eval() if fuse and hasattr(ckpt, "fuse") else ckpt.eval())  # model in eval mode

    # Module updates
    for m in model.modules():
        t = type(m)
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model):
            m.inplace = inplace
            if t is Detect and not isinstance(m.anchor_grid, list):
                delattr(m, "anchor_grid")
                setattr(m, "anchor_grid", [torch.zeros(1)] * m.nl)
        elif t is nn.Upsample and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model
    if len(model) == 1:
        return model[-1]

    # Return detection ensemble
    print(f"Ensemble created with {weights}\n")
    for k in "names", "nc", "yaml":
        setattr(model, k, getattr(model[0], k))
    model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model])).int()].stride  # max stride
    assert all(model[0].nc == m.nc for m in model), f"Models have different class counts: {[m.nc for m in model]}"
    return model

    # """
    # Loads and fuses an ensemble or single YOLOv5 model from weights, handling device placement and model adjustments.
    # """
    # from models.yolo import Detect, Model
    #
    # model = Ensemble()
    # for w in weights if isinstance(weights, list) else [weights]:
    #     ckpt = torch.load(attempt_download(w), map_location="cpu")  # load
    #
    #     # Check if the checkpoint contains a model or ema state dictionary
    #     if 'model' in ckpt:
    #         model_state_dict = ckpt['model'].state_dict()
    #     elif 'ema' in ckpt:
    #         model_state_dict = ckpt['ema'].state_dict()
    #     else:
    #         raise KeyError("'model' or 'ema' key not found in checkpoint. Available keys:", list(ckpt.keys()))
    #
    #     # Ensure that model_state_dict is a dictionary
    #     if not isinstance(model_state_dict, dict):
    #         raise TypeError("Expected state_dict to be dict-like, got {}".format(type(model_state_dict)))
    #
    #     # Create model
    #     from models.yolo import Model  # 假设这是模型定义的位置
    #     default_cfg = 'models/yolov5s.yaml'  # 默认配置文件路径
    #     model_instance = Model(default_cfg)  # 使用默认配置文件创建模型
    #     model_instance.load_state_dict(model_state_dict, strict=False)  # load state_dict
    #
    #     # Model compatibility updates
    #     if not hasattr(model_instance, "stride"):
    #         model_instance.stride = torch.tensor([32.0])
    #     if hasattr(model_instance, "names") and isinstance(model_instance.names, (list, tuple)):
    #         model_instance.names = dict(enumerate(model_instance.names))  # convert to dict
    #
    #     model.append(model_instance.fuse().eval() if fuse and hasattr(model_instance,
    #                                                                   "fuse") else model_instance.eval())  # model in eval mode
    #
    # # Module updates
    # for m in model.modules():
    #     t = type(m)
    #     if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model):
    #         m.inplace = inplace
    #         if t is Detect and not isinstance(m.anchor_grid, list):
    #             delattr(m, "anchor_grid")
    #             setattr(m, "anchor_grid", [torch.zeros(1)] * m.nl)
    #     elif t is nn.Upsample and not hasattr(m, "recompute_scale_factor"):
    #         m.recompute_scale_factor = None  # torch 1.11.0 compatibility
    #
    # # Return model
    # if len(model) == 1:
    #     return model[-1]
    #
    # # Return detection ensemble
    # print(f"Ensemble created with {weights}\n")
    # for k in "names", "nc", "yaml":
    #     setattr(model, k, getattr(model[0], k))
    # model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model])).int()].stride  # max stride
    # assert all(model[0].nc == m.nc for m in model), f"Models have different class counts: {[m.nc for m in model]}"
    # return model