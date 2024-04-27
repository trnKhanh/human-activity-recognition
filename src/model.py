import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import drop_path

import numpy as np


class DropPath(nn.Module):
    def __init__(self, drop_prob):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        out_channels=768,
        frame_size=16,
        tube_size=2,
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.tube_size = tube_size
        self.num_patches = (frame_size // tube_size) * (
            img_size // patch_size
        ) ** 2

        self.proj = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(tube_size, patch_size, patch_size),
            stride=(tube_size, patch_size, patch_size),
        )

    def forward(self, x):
        B, C, T, H, W = x.shape
        assert (
            H == self.img_size and W == self.img_size
        ), f"Input image size ({H}x{W}) does not match model ({self.img_size}x{self.img_size})"
        # output_shape: B, N, C
        output = self.proj(x).flatten(2).permute(0, 2, 1)
        output = output.contiguous()
        return output


class Attention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads=8,
        have_qkv_bias=False,
        attn_head_dim=None,
        attn_drop_rate=0.0,
        proj_drop_rate=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads

        head_dim = embed_dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim

        all_head_dims = head_dim * self.num_heads

        self.qkv_proj = nn.Linear(embed_dim, all_head_dims * 3, bias=False)
        if have_qkv_bias:
            self.qkv_bias = nn.Parameter(torch.zeros(all_head_dims * 3))
        else:
            self.qkv_bias = None

        self.attn_drop = nn.Dropout(attn_drop_rate)

        self.output_proj = nn.Linear(all_head_dims, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop_rate)

    def forward(self, x):
        B, N, C = x.shape

        qkv = F.linear(x, weight=self.qkv_proj.weight, bias=self.qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, -1)

        x = self.output_proj(x)
        x = self.proj_drop(x)

        return x


class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop_rate=0.0,
    ):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads=8,
        have_qkv_bias=False,
        attn_head_dim=None,
        attn_drop_rate=0.0,
        mlp_ratio=4,
        drop_rate=0.0,
        drop_path_rate=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm_1 = norm_layer(embed_dim)
        self.attn = Attention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            have_qkv_bias=have_qkv_bias,
            attn_head_dim=attn_head_dim,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=drop_rate,
        )
        self.norm_2 = norm_layer(embed_dim)
        self.mlp = MLP(
            in_features=embed_dim,
            hidden_features=embed_dim * mlp_ratio,
            out_features=embed_dim,
            act_layer=act_layer,
            drop_rate=drop_rate,
        )
        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        )

    def forward(self, x):
        B, N, C = x.shape
        x = x + self.drop_path(self.attn(self.norm_1(x)))
        x = x + self.drop_path(self.mlp(self.norm_2(x)))

        return x


def get_positional_encoding(num_pos, num_dims):
    def get_angle_vector(i):
        return [
            i / np.power(10000, 2 * ((j // 2) / num_dims))
            for j in range(num_dims)
        ]

    table = np.array([get_angle_vector(i) for i in range(num_pos)])
    table[:, 0::2] = np.sin(table[:, 0::2])
    table[:, 1::2] = np.cos(table[:, 1::2])

    table = torch.from_numpy(table)
    table = table.unsqueeze(0)
    return table


class VIT(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        num_classes=1000,
        frame_size=16,
        tube_size=2,
        depth=12,
        num_heads=8,
        have_qkv_bias=False,
        attn_head_dim=None,
        attn_drop_rate=0.0,
        mlp_ratio=4,
        drop_rate=0.0,
        drop_path_rate=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        head_drop_rate=0.0,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            frame_size=frame_size,
            tube_size=tube_size,
            in_channels=in_channels,
            out_channels=embed_dim,
        )
        self.num_patches = self.patch_embed.num_patches
        self.pos_encoding = get_positional_encoding(self.num_patches, embed_dim)
        drop_path_rates = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    have_qkv_bias=have_qkv_bias,
                    attn_head_dim=attn_head_dim,
                    attn_drop_rate=attn_drop_rate,
                    mlp_ratio=mlp_ratio,
                    drop_rate=drop_rate,
                    drop_path_rate=drop_path_rates[i],
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        self.head_drop = nn.Dropout(head_drop_rate)
        self.fc_norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = self.patch_embed(x)
        B, N, C = x.shape

        x = x + self.pos_encoding.type_as(x).to(x.device)

        for blk in self.blocks:
            x = blk(x)

        x = self.fc_norm(x.mean(1))
        x = self.head(self.head_drop(x))
        return x


def stvit_base_patch16_224(**kwargs):
    model = VIT(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        frame_size=16,
        tube_size=2,
        depth=12,
        num_heads=8,
        have_qkv_bias=False,
        mlp_ratio=4,
        **kwargs,
    )

    return model
