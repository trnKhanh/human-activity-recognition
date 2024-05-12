import torch
from torch import nn

from models.utils import build_A
from .layers import Block
from .graph import NTUGraph


class STGCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int = 60,
        act_layer=nn.ReLU,
        dropout_rate: float = 0,
    ):
        super().__init__()
        self.graph = NTUGraph()
        self.data_norm = nn.BatchNorm1d(
            in_channels * self.graph.get_num_joints()
        )
        layer_cfs = [
            (in_channels, 48, 1),
            (48, 48, 1),
            (48, 48, 1),
            (48, 48, 1),
            (48, 96, 2),
            (96, 96, 1),
            (96, 96, 1),
            (96, 192, 2),
            (192, 192, 1),
            (192, 192, 1),
        ]
        down_id = {
            4: 1, 
            7: 2,}
        self.blocks = nn.ModuleList()
        A = self.graph.get_decompose(0)
        for i, cf in enumerate(layer_cfs):
            if i in down_id:
                A = self.graph.get_decompose(down_id[i])

            self.blocks.append(
                Block(
                    cf[0],
                    cf[1],
                    A[0] if i in down_id else A[1],
                    stride=cf[2],
                    dropout_rate=dropout_rate,
                    residual=(i > 0 and i not in down_id),
                    act_layer=act_layer,
                    init_block=(i == 0),
                )
            )
        self.avg_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.head = nn.Linear(layer_cfs[-1][1], num_classes)

    def forward(self, x):
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_norm(x)
        x = x.view(N * M, V, C, T)
        x = x.permute(0, 2, 3, 1).contiguous()

        for block in self.blocks:
            x = block(x)

        NM, C, T, V = x.size()
        x = x.view(N, M, C, T, V)
        x = x.permute(0, 2, 3, 1, 4).contiguous()
        x = x.view(N, C, T, V * M)
        x = torch.squeeze(self.avg_pool(x), dim=(2, 3))
        x = self.head(x)
        return x
