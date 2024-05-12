import torch
from torch import nn
from .graph import NTUGraph


class TCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        branch_cf=[(3, 1), (3, 2), (3, 3), (3, 4), ("max", 3), ("1x1")],
        stride: int = 1,
        dropout_rate: float = 0,
        act_layer=nn.ReLU,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.num_branches = len(branch_cf)
        self.dropout = (
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        )

        self.act = act_layer() if act_layer is not None else nn.Identity()

        self.branch_channels = out_channels // self.num_branches
        self.rem_branch_channels = out_channels - self.branch_channels * (
            self.num_branches - 1
        )

        branches = []
        for i, cf in enumerate(branch_cf):
            branch_c = (
                self.rem_branch_channels if i == 0 else self.branch_channels
            )
            if cf == "1x1":
                branches.append(
                    nn.Conv2d(
                        in_channels,
                        branch_c,
                        kernel_size=1,
                        stride=(stride, 1),
                    )
                )
            elif cf[0] == "max":
                padding = cf[1] // 2
                branches.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels,
                            branch_c,
                            kernel_size=1,
                        ),
                        nn.BatchNorm2d(branch_c),
                        self.act,
                        self.dropout,
                        nn.MaxPool2d(
                            kernel_size=(cf[1], 1),
                            stride=(stride, 1),
                            padding=(padding, 0),
                        ),
                    )
                )
            else:
                padding = (cf[0] + (cf[0] - 1) * (cf[1] - 1) - 1) // 2
                branches.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, branch_c, kernel_size=1),
                        nn.BatchNorm2d(branch_c),
                        self.act,
                        self.dropout,
                        nn.Conv2d(
                            branch_c,
                            branch_c,
                            kernel_size=(cf[0], 1),
                            stride=(stride, 1),
                            dilation=(cf[1], 1),
                            padding=(padding, 0),
                        ),
                    )
                )
        self.branches = nn.ModuleList(branches)

    def forward(self, x):
        N, C, T, V = x.size()
        branch_outs = []
        for branch in self.branches:
            branch_outs.append(branch(x))

        y = torch.cat(branch_outs, dim=1)
        return y


class GCNUnit(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A,
        adaptive=None,
        embed_coef: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = (out_channels,)

        # self.A = A.clone()
        self.adaptive = adaptive
        self.register_buffer("A", A.clone())

        if adaptive == "init":
            self.PA = nn.Parameter(A.clone())
        elif adaptive == "offset":
            self.PA = nn.Parameter(torch.zeros(A.size()))
            nn.init.uniform_(self.PA, -1e-6, 1e-6)
        elif adaptive == "importance":
            self.PA = nn.Parameter(torch.ones(A.size()))
        elif adaptive == "data-driven":
            self.PA = nn.Parameter(A.clone())
            self.alpha = nn.Parameter(torch.zeros(1))

            self.avg_pool = nn.AdaptiveAvgPool2d((1, None))
            self.inter_channels = out_channels // embed_coef
            self.convA1 = nn.Conv2d(in_channels, self.inter_channels, 1)
            self.convA2 = nn.Conv2d(in_channels, self.inter_channels, 1)

        self.conv = nn.Conv2d(in_channels, out_channels * A.size(0), 1)

    def forward(self, x):
        N, C, T, V = x.size()

        if self.adaptive == "init":
            A = self.PA
        elif self.adaptive == "offset":
            A = self.A + self.PA
        elif self.adaptive == "importance":
            A = self.A * self.PA
        elif self.adaptive == "data-driven":
            y = self.avg_pool(x)
            A1 = torch.squeeze(self.convA1(y), dim=2)
            A2 = torch.squeeze(self.convA2(y), dim=2)
            A1 = A1.permute(0, 2, 1).contiguous()  # N, C, V => N, V, C
            A_data = torch.softmax(torch.matmul(A1, A2), dim=2)
            A = self.PA + self.alpha * A_data
        else:
            A = self.A

        x = self.conv(x)
        x = x.view(N, self.A.size(0), -1, T, V)
        x = torch.einsum("nkctv,kvw->nctw", (x, A)).contiguous()
        return x


class GCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        branch_cf,
        concat=False,
        dropout_rate: float = 0,
        act_layer=nn.ReLU,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = (out_channels,)

        self.num_branches = len(branch_cf)

        self.dropout = (
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        )
        self.act = act_layer() if act_layer is not None else nn.Identity()

        self.branch_channels = (
            out_channels // self.num_branches if concat else out_channels
        )
        self.rem_branch_channels = (
            out_channels - self.branch_channels * (self.num_branches - 1)
            if concat
            else out_channels
        )
        self.branches = nn.ModuleList()
        for i, cf in enumerate(branch_cf):
            branch_c = (
                self.rem_branch_channels if i == 0 else self.branch_channels
            )
            self.branches.append(
                nn.Sequential(
                    GCNUnit(
                        in_channels,
                        branch_c,
                        A=cf[1],
                        adaptive=cf[0],
                    ),
                )
            )
        self.concat = concat

    def forward(self, x):
        N, C, T, V = x.size()

        branch_outs = []
        for branch in self.branches:
            branch_outs.append(branch(x))

        if self.concat:
            y = torch.cat(branch_outs, dim=1)
        else:
            y = torch.stack(branch_outs).sum(dim=0)
        return y


class Block(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A,
        stride: int = 1,
        dropout_rate: float = 0,
        residual: bool = True,
        act_layer=nn.ReLU,
        init_block=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        gcn_branch_cf = [
            ("offset", A),
        ]

        self.gcn = GCN(
            in_channels,
            out_channels,
            branch_cf=gcn_branch_cf,
            dropout_rate=dropout_rate,
            act_layer=act_layer,
        )
        if not residual:
            self.g_res = lambda x: 0
        elif in_channels == out_channels:
            self.g_res = lambda x: x
        else:
            self.g_res = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
            )
        self.g_norm = nn.BatchNorm2d(out_channels)

        tcn_branch_cf = [(5, 1), (5, 2), ("max", 3), ("1x1")]
        self.tcn = TCN(
            out_channels,
            out_channels,
            branch_cf=tcn_branch_cf,
            stride=stride,
            dropout_rate=dropout_rate,
            act_layer=act_layer,
        )
        if not residual:
            self.t_res = lambda x: 0
        elif out_channels == out_channels and stride == 1:
            self.t_res = lambda x: x
        else:
            self.t_res = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        self.t_norm = nn.BatchNorm2d(out_channels)
        self.act = act_layer()
        self.dropout = (
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        )

    def forward(self, x):
        g_res = self.g_res(x)
        x = self.gcn(x)
        x = self.act(self.g_norm(x) + g_res)
        x = self.dropout(x)

        t_res = self.t_res(x)
        x = self.tcn(x)
        x = self.act(self.t_norm(x) + t_res)
        x = self.dropout(x)

        return x
