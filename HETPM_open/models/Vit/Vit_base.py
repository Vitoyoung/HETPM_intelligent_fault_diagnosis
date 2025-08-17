import math
from functools import partial
from itertools import repeat
import torch
import torch.nn as nn
import torch.nn.functional as F

import collections.abc as container_abcs


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# 两个FC    用于Feed Forward
class Mlp(nn.Module):
    def __init__(self, in_features, drop, hidden_features=None, out_features=None, act_layer=nn.GELU ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



# MSA    (q*k) * v  之后投影
class Attention(nn.Module):
    def __init__(self, dim, attn_drop, proj_drop, num_heads=5, qkv_bias=False, qk_scale=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads # 50
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn = None


    def forward(self, x, use_attn=True):
        #[8, 21, 50]
        B, N, C = x.shape
        #如果是预测

        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        self.attn = attn
        attn2 = self.attn_drop(attn)
        x = (attn2 @ v) if use_attn else v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x



class Block(nn.Module):

    def __init__(self, dim, num_heads, drop, attn_drop, drop_path, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        # Feed Forward
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    # x
    def forward(self, x):

        x = self.attn(self.norm1(x))
        x = x + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x



def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        print("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):

    return _no_grad_trunc_normal_(tensor, mean, std, a, b)





#
class Patch_Embed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, patch_size, signal_size, in_chans):
        super().__init__()
        # patch数量   4个patch，每个patch大小为  1*256
        self.num_patches = signal_size // patch_size
        self.proj = nn.Conv1d(in_channels=in_chans, out_channels=patch_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        #B, C, H = x.shape
        # B*1*1024  >>  B*50*20
        x = self.proj(x)
        x = x.transpose(1, 2)   # [B, 20, 50]
        return x




class Vit_base_1d(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, signal_size=1024, embed_dim=32, in_channel=1, depth=1, num_heads=32, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        # ************************************************************************************
        # 输入图片打成patch
        self.patch_embed = Patch_Embed(patch_size=embed_dim, signal_size=signal_size, in_chans=in_channel)
        num_patches = self.patch_embed.num_patches
        # 分类令牌
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # 绝对位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 每个Block_3_branches层逐渐增加的dropout rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # 多个Block_3_branches堆叠
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                             qk_scale=qk_scale,drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                             norm_layer=norm_layer)for i in range(depth)])

        self.norm = norm_layer(embed_dim)
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # x  >>  B*1*1024
    def forward_features(self, x):

        # batchsize
        B = x.shape[0]
        # B*20*50
        x = self.patch_embed(x)
        # B*1*50
        cls_tokens = self.cls_token.expand(B, -1, -1)
        # 拼接分类令牌
        x = torch.cat((cls_tokens, x), dim=1)  # B*21*50
        # summation位置编码
        x = x + self.pos_embed
        # 过 dropout
        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            x = blk(x)

        x = self.norm(x)
        # 返回的是  源域和目标域 的分类令牌列
        return x



    def forward(self, x):
        x = self.forward_features(x)
        return x



def Vit_base_features(**kwargs):

    model = Vit_base_1d()

    return model



class Vit_base_features_1d(nn.Module):
    def __init__(self):       #
        super(Vit_base_features_1d, self).__init__()
        self.Vit_base_features = Vit_base_features()
        self.__in_features = Vit_base_features().embed_dim

    def forward(self, x):
        x = self.Vit_base_features(x)
        return x[:, 0, :]

    def output_num(self):
        return self.__in_features




class Vit_base_features_4096_1d(nn.Module):
    def __init__(self):       #
        super(Vit_base_features_4096_1d, self).__init__()
        self.Vit_base_features = Vit_base_features()
        self.__in_features = 4096
        self.proj = nn.Linear(32, 4096)

    def forward(self, x):
        x = self.Vit_base_features(x)
        x = self.proj(x[:, 0, :])
        return x

    def output_num(self):
        return self.__in_features



# Transfer = Vit_base_features_1d(pretrained = False)
# transfer_src_input = torch.rand(8, 1, 1024)
# transfer_tgt_input = torch.rand(8, 1, 1024)
# inference_target = True
# out = Transfer(transfer_src_input)
# print(out.size())
# print(len(out))

