import torch
from torch import nn

from models.utils import build_A


class GCNUnit(nn.Module):
    def __init__(self, in_channels, out_channels, A, importance=True):
        super().__init__()
        self.A = A
        self.conv = nn.Conv2d(
            in_channels, out_channels * A.size()[0], kernel_size=(1, 1)
        )
        self.importance = (
            nn.Parameter(torch.ones(A.size()))
            if importance
            else torch.tensor(1.0)
        )
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        N, C, T, V = x.size()
        x = self.conv(x)

        N, KC, T, V = x.size()
        x = x.view(N, self.A.size(0), -1, T, V)
        x = torch.einsum(
            "nkctv,kvw->nctw", (x, self.A * self.importance)
        ).contiguous()
        x = self.norm(x)
        return x


class TCNUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        padding = (kernel_size // 2, 0)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            stride=(stride, 1),
            padding=padding,
        )
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.norm(self.conv(x))
        return x


class STGCNBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        A,
        importance=True,
        t_kernel_size=9,
        t_stride=1,
        dropout_rate=0.0,
        residual=True,
        act_layer=nn.GELU,
    ):
        super().__init__()
        self.gcn = GCNUnit(in_channels, out_channels, A, importance)
        self.act = act_layer()
        self.tcn = TCNUnit(out_channels, out_channels, t_kernel_size, t_stride)
        self.drop = nn.Dropout(dropout_rate)

        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and t_stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=(1, 1),
                    stride=(t_stride, 1),
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        res = self.residual(x)
        x = self.act(self.gcn(x))
        x = self.drop(self.tcn(x))
        x = self.act(x + res)

        return x


class STGCN(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes,
        A,
        importance=True,
        residual=True,
        dropout_rate=0.0,
        act_layer=nn.GELU,
    ):
        super().__init__()
        self.A = A
        self.data_norm = nn.BatchNorm1d(in_channels * self.A.size(1))

        t_kernel_size = 9
        ins = [in_channels, 64, 64, 64, 64, 128, 128, 128, 256, 256]
        outs = [64, 64, 64, 64, 128, 128, 128, 256, 256, 256]
        strides = [1, 1, 1, 1, 2, 1, 1, 2, 1, 1]

        self.blocks = nn.ModuleList(
            [
                STGCNBlock(
                    ins[i],
                    outs[i],
                    A=self.A,
                    importance=importance,
                    t_kernel_size=t_kernel_size,
                    t_stride=strides[i],
                    dropout_rate=dropout_rate,
                    residual=residual,
                    act_layer=act_layer,
                )
                for i in range(10)
            ]
        )
        self.avg_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(outs[-1], num_classes)

    def forward(self, x):
        N, C, T, V, M = x.size()
        x = x.permute(
            0, 4, 3, 1, 2
        ).contiguous()  # N, C, T, V, M => N, M, V, C, T
        x = x.view(N * M, V * C, T)
        x = self.data_norm(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(
            0, 1, 3, 4, 2
        ).contiguous()  # N, M, V, C, T => N, M, C, T, V
        x = x.view(N * M, C, T, V)

        for blk in self.blocks:
            x = blk(x)
        NM, C, T, V = x.size()
        x = x.view(N, M, C, T, V)
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = x.view(N, C, T, V * M)
        x = self.avg_pool(x).squeeze(dim=(2, 3))
        x = self.fc(x)
        x = x.softmax(-1)
        return x
