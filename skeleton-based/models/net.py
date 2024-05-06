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
            (in_channels, 64, 1),
            (64, 64, 1),
            (64, 64, 1),
            (64, 64, 1),
            (64, 128, 2),
            (128, 128, 1),
            (128, 128, 1),
            (128, 256, 2),
            (256, 256, 1),
            (256, 256, 1),
        ]
        self.blocks = nn.ModuleList()
        for i, cf in enumerate(layer_cfs):
            self.blocks.append(
                Block(
                    cf[0],
                    cf[1],
                    self.graph,
                    stride=cf[2],
                    dropout_rate=dropout_rate,
                    residual=(i > 0),
                    act_layer=act_layer,
                    init_block=(i == 0),
                )
            )
        self.avg_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.head = nn.Linear(256, num_classes)

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
