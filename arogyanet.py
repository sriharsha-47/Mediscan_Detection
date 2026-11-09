"""ArogyaNet: a small, from-scratch CNN for 1-channel chest X-rays (< 2 M parameters, no pretrained weights).

Deliberate design decisions
  D1  residual (skip) connections around every block          -> use_res
  D2  squeeze-and-excitation channel attention in every block -> use_se
  (also: depthwise-separable convs for CPU speed, global-average-pool head instead of flatten+dense)
Two-branch variant (use_meta=True): 3 handcrafted numbers -> small dense branch, concatenated before the classifier.
"""
import torch, torch.nn as nn


def conv_bn(cin, cout, k, s=1, groups=1, act=True):
    layers = [nn.Conv2d(cin, cout, k, s, k // 2, groups=groups, bias=False), nn.BatchNorm2d(cout)]
    if act: layers.append(nn.ReLU6(inplace=True))
    return nn.Sequential(*layers)


class SE(nn.Module):
    def __init__(self, c, r=8):
        super().__init__(); m = max(c // r, 4)
        self.fc = nn.Sequential(nn.Linear(c, m), nn.ReLU(inplace=True), nn.Linear(m, c), nn.Sigmoid())
    def forward(self, x):
        return x * self.fc(x.mean((2, 3)))[:, :, None, None]


class Block(nn.Module):
    """2 x depthwise-separable conv  ->  (SE)  ->  (+ shortcut)  -> ReLU6"""
    def __init__(self, cin, cout, stride, use_se, use_res):
        super().__init__()
        self.dw1 = conv_bn(cin, cin, 3, stride, groups=cin); self.pw1 = conv_bn(cin, cout, 1)
        self.dw2 = conv_bn(cout, cout, 3, 1, groups=cout);   self.pw2 = conv_bn(cout, cout, 1, act=False)
        self.se = SE(cout) if use_se else nn.Identity()
        self.use_res = use_res
        if use_res:
            if stride != 1 or cin != cout:
                self.short = nn.Sequential(nn.AvgPool2d(stride) if stride > 1 else nn.Identity(),
                                           nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout))
            else:
                self.short = nn.Identity()
        self.act = nn.ReLU6(inplace=True)
    def forward(self, x):
        out = self.se(self.pw2(self.dw2(self.pw1(self.dw1(x)))))
        if self.use_res: out = out + self.short(x)
        return self.act(out)


class ArogyaNet(nn.Module):
    def __init__(self, use_se=True, use_res=True, use_meta=False, dropout=0.2, width=(16, 32, 64, 128, 192), meta_hidden=16):
        super().__init__()
        self.use_meta = use_meta
        self.norm = nn.BatchNorm2d(1)
        self.stem = conv_bn(1, width[0], 3, 2)
        chans = list(width)
        self.block1 = Block(chans[0], chans[1], 2, use_se, use_res)
        self.block2 = Block(chans[1], chans[2], 2, use_se, use_res)
        self.block3 = Block(chans[2], chans[3], 2, use_se, use_res)
        self.block4 = Block(chans[3], chans[4], 2, use_se, use_res)
        self.block5 = Block(chans[4], chans[4], 1, use_se, use_res)
        self.gap = nn.AdaptiveAvgPool2d(1)
        feat = chans[4]
        if use_meta:
            self.meta = nn.Sequential(nn.BatchNorm1d(3), nn.Linear(3, meta_hidden), nn.ReLU(inplace=True),
                                      nn.Linear(meta_hidden, meta_hidden), nn.ReLU(inplace=True))
            feat += meta_hidden
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(feat, 1)

    def embed(self, x, meta=None):
        x = self.block5(self.block4(self.block3(self.block2(self.block1(self.stem(self.norm(x)))))))
        z = self.gap(x).flatten(1)
        if self.use_meta: z = torch.cat([z, self.meta(meta)], 1)
        return z                                           # penultimate-layer embedding

    def forward(self, x, meta=None):
        return self.fc(self.drop(self.embed(x, meta))).squeeze(1)


class Transfer(nn.Module):
    """ImageNet-pretrained backbone; the 1-channel X-ray is repeated to 3 channels."""
    def __init__(self, name, dropout=0.2, pretrained=True):
        super().__init__()
        import torchvision.models as tv
        try:
            if name == "resnet18":
                m = tv.resnet18(weights=tv.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
                feat = m.fc.in_features; m.fc = nn.Identity()
            else:
                m = tv.mobilenet_v3_small(weights=tv.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
                feat = m.classifier[0].in_features; m.classifier = nn.Identity()
        except Exception as e:
            raise RuntimeError(f"Could not load pretrained weights ({e}). Connect to the internet once so torchvision can "
                               f"download them, or pass --no_pretrained (and report that you did).")
        self.backbone, self.drop, self.fc = m, nn.Dropout(dropout), nn.Linear(feat, 1)
    def embed(self, x, meta=None):
        x = ((x - 0.449) / 0.226).repeat(1, 3, 1, 1)
        return self.backbone(x)
    def forward(self, x, meta=None):
        return self.fc(self.drop(self.embed(x))).squeeze(1)


def build_model(name, dropout=0.2, pretrained=True):
    if name == "arogyanet":       return ArogyaNet(dropout=dropout)
    if name == "arogyanet_nores": return ArogyaNet(use_res=False, dropout=dropout)
    if name == "arogyanet_nose":  return ArogyaNet(use_se=False, dropout=dropout)
    if name == "arogyanet_meta":  return ArogyaNet(use_meta=True, dropout=dropout)
    if name in ("resnet18", "mobilenetv3"): return Transfer(name, dropout, pretrained)
    raise ValueError(name)


def count_params(m): return sum(p.numel() for p in m.parameters())
