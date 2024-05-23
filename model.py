import math

from torch import nn

from hyps import Hyps


class Conv(nn.Module):
    def __init__(self, cin, cout, ksize, stride=1, norm=True, bias=False, drop=0.0, act=True):
        super().__init__()
        ksize = int(math.ceil((ksize - 1) / 2)) * 2 + 1
        self.conv = nn.Conv2d(cin, cout, ksize, stride, bias=bias, padding=(ksize - 1) // 2)
        self.drop = nn.Dropout(drop) if drop > 0 else None
        self.norm = nn.BatchNorm2d(cout) if norm else None
        self.act = nn.ReLU() if act else None

    def forward(self, x):
        x = self.conv(x)
        if self.drop:
            x = self.drop(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class ArchBlock(nn.Module):
    def __init__(self, hyps: Hyps, cin, cout, ksize, stride=1):
        super().__init__()
        cmid = int(math.ceil(hyps.bottleneck * cout))

        self.conv1 = Conv(cin, cmid, 1, drop=hyps.dropout)
        self.conv2 = Conv(cmid, cmid, ksize, stride=stride, drop=hyps.dropout)
        self.conv3 = Conv(cmid, cout, 1, drop=hyps.dropout, act=False)

        self.conv_res = Conv(cin, cout, 1, stride=stride, act=False) if cin != cout or stride != 1 else None

        self.act = nn.ReLU()

    def forward(self, x):
        x0 = self.conv_res(x) if self.conv_res else x

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x + x0
        x = self.act(x)
        return x


class Stage(nn.Module):
    def __init__(self, hyps: Hyps, cin, cmid, n_blocks):
        super().__init__()

        self.blocks = nn.ModuleList([
            ArchBlock(hyps, cin, cmid, hyps.main_ksize, stride=2)
        ] + [
            ArchBlock(hyps, cmid, cmid, hyps.main_ksize)
            for _ in range(n_blocks)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class Stem(nn.Module):
    def __init__(self, hyps: Hyps):
        super().__init__()

        self.conv = Conv(3, hyps.stem_feats, hyps.stem_ksize, stride=2 if hyps.stem_type == "conv" else 1)
        self.pool = nn.MaxPool2d(2) if hyps.stem_type == "max_pool" else None

    def forward(self, x):
        x = self.conv(x)
        if self.pool:
            x = self.pool(x)
        return x


class Head(nn.Module):
    def __init__(self, hyps: Hyps, feat_size: int, n_classes: int):
        super().__init__()

        self.pool = {
            "avg_pool": nn.Sequential(
                Conv(feat_size, hyps.head_hid_dim, ksize=1),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten()
            ),
            "dense": nn.Sequential(
                nn.Flatten(),
                nn.LazyLinear(hyps.head_hid_dim)
            )
        }[hyps.head_type]

        self.pred = nn.Linear(hyps.head_hid_dim, n_classes)

    def forward(self, x):
        x = self.pool(x)
        x = self.pred(x)
        return x


class Classifier(nn.Module):
    def __init__(self, hyps: Hyps, n_classes: int):
        super().__init__()

        self.stem = Stem(hyps)

        self.stages = nn.ModuleList([
            Stage(
                hyps,
                int(math.ceil(hyps.stem_feats * hyps.feats_gain ** (i - 1 if i >= 1 else 0))),
                int(math.ceil(hyps.stem_feats * hyps.feats_gain ** (i))),
                n_blocks=hyps.lvl_blocks
            )
            for i in range(hyps.lvl_count)
        ])
        feats = int(math.ceil(hyps.stem_feats * hyps.feats_gain ** (hyps.lvl_count - 1)))

        self.head = Head(hyps, feats, n_classes)

    def forward(self, x):
        x = self.stem(x)

        for stage in self.stages:
            x = stage(x)

        x = self.head(x)

        return x

