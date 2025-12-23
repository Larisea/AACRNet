"""
 @Time    : 2024/10/6 14:23
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile, clever_format
from torchvision.models import resnet18

# from ECANet, in which y and b is set default to 2 and 1
def kernel_size(in_channel):
    """Compute kernel size for one dimension convolution in eca-net"""
    k = int((math.log2(in_channel) + 1) // 2)  # parameters from ECA-net
    if k % 2 == 0:
        return k + 1
    else:
        return k

###################################################################
# ######################## 上下文探索 Module ###########################
###################################################################

class CE(nn.Module):
    def __init__(self, dim, k_size=3, d_list=[1, 2, 3, 4]):
        super().__init__()
        group_size = dim // 4  # 每组的通道数

        # 分组卷积（每组用不同的膨胀率）
        self.g0 = self._build_group_conv(group_size, k_size, d_list[0])
        self.g1 = self._build_group_conv(group_size, k_size, d_list[1])
        self.g2 = self._build_group_conv(group_size, k_size, d_list[2])
        self.g3 = self._build_group_conv(group_size, k_size, d_list[3])

        # 1x1卷积，用于整合通道信息
        self.tail_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
        )
        self.conv = nn.Conv2d(dim, dim//4, kernel_size=1)

    def _build_group_conv(self, group_size, k_size, dilation):
        """构建分组卷积"""
        return nn.Sequential(
            nn.Conv2d(group_size, group_size, kernel_size=k_size, stride=1, padding=(k_size + (k_size - 1) * (dilation - 1)) // 2, dilation=dilation,groups=group_size),
            nn.BatchNorm2d(group_size),
            nn.ReLU()
        )

    def forward(self, x, log=False, module_name=None, img_name=None):
        identity = x
        # 将输入特征按通道拆分为4组
        xs = self.conv(x)
        x0 = self.g0(xs)
        x1 = self.g1(xs)
        x2 = self.g2(xs)
        x3 = self.g3(xs)
        # 拼接4个分组的输出
        x = torch.cat((x0, x1, x2, x3), dim=1)
        # x = self.tail_conv(x)
        x = self.tail_conv(x) + identity
        return x


def dsconv_3x3(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, groups=in_channel),
        nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, groups=1),
        nn.BatchNorm2d(out_channel),
        nn.ReLU6(inplace=True)
    )


###################################################################
# ######################## Focus Module ###########################
###################################################################
class Focus(nn.Module):
    def __init__(self, channel1, channel2):
        super(Focus, self).__init__()
        self.channel1 = channel1
        self.channel2 = channel2
        self.up = nn.Sequential(dsconv_3x3(self.channel2, self.channel1)
                                )
        self.input_map = nn.Sigmoid()
        self.output_map = nn.Conv2d(self.channel1, 1, 1)
        self.convx = dsconv_3x3(channel1, channel1)
        self.fp = CE(self.channel1)
        self.fn = CE(self.channel1)
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.bn1 = nn.BatchNorm2d(self.channel1)
        self.relu1 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(self.channel1)
        self.relu2 = nn.ReLU()

    def forward(self, x, y, in_map, log=False, module_name=None, img_name=None):
        # x: 当前层的特征图
        # y: 高层次特征图
        # in_map: 高层次预测掩码
        up = F.interpolate(y, size=x.size()[2:], mode='bilinear', align_corners=True)
        up = self.up(up)
        in_map = F.interpolate(in_map, size=x.size()[2:], mode='bilinear', align_corners=True)
        # 控制 input_map 的权重值
        input_map = self.input_map(in_map)
        # 前景和背景特征分离
        f_feature = x * (input_map)
        b_feature = x * (1 - input_map)

        # 上下文特征提取
        if log:
            # 假阳性
            fp = self.fp(f_feature, log=log, module_name=module_name + '-fp_ce', img_name=img_name)
            # 假阴性
            fn = self.fn(b_feature, log=log, module_name=module_name + '-fn_ce', img_name=img_name)
        else:
            # 假阳性
            fp = self.fp(f_feature)
            # 假阴性
            fn = self.fn(b_feature)
        # 消除背景干扰
        refine1 = up - (self.alpha * fp)
        refine1 = self.bn1(refine1)
        refine1 = self.relu1(refine1)

        # 从背景补充前景信息
        refine2 = refine1 + (self.beta * fn)
        refine2 = self.bn2(refine2)
        refine2 = self.relu2(refine2)

        # 输出结果图
        output_map = self.output_map(refine2)

        return refine2, output_map


class ResNet(torch.nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.resnet = resnet18(pretrained=True, replace_stride_with_dilation=[False, False, False])

    def forward(self, x):
        # resnet layers
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)

        x_2 = self.resnet.layer1(x)  # size->1/2, in=64, out=64
        x_4 = self.resnet.layer2(x_2)  # size-> 1/4, in=64, out=128
        x_8 = self.resnet.layer3(x_4)  # size-> 1/8, in=128, out=256
        x_16 = self.resnet.layer4(x_8)  # size-> 1/16, in=256, out=512

        return x_2, x_4, x_8, x_16


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class SAM(nn.Module):
    def __init__(self, channel=64):
        super(SAM, self).__init__()
        self.Translayer_2 = nn.Conv2d(128, channel, 1)
        self.Translayer_3 = nn.Conv2d(256, channel, 1)
        self.Translayer_4 = nn.Conv2d(512, channel, 1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 1)
        self.conv4 = BasicConv2d(3 * channel, channel, 1)
        self.conv5 = nn.Conv2d(channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1 = self.Translayer_4(x1)
        x2 = self.Translayer_3(x2)
        x3 = self.Translayer_2(x3)
        # 语义过滤
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample3(self.upsample(x2_1)) * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.upsample(x1_1)), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.upsample(x2_2)), 1)
        x3_2 = self.conv_concat3(x3_2)

        sematic = self.conv4(x3_2)
        coarse = self.conv5(sematic)
        return sematic, coarse


class TDFM(nn.Module):
    def __init__(self, in_channel):
        super(TDFM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # ECA的部分
        self.k = kernel_size(in_channel)
        self.channel_conv1 = nn.Conv1d(4, 1, kernel_size=self.k, padding=self.k // 2)
        self.channel_conv2 = nn.Conv1d(4, 1, kernel_size=self.k, padding=self.k // 2)
        self.diffw = nn.Sequential(
            dsconv_3x3(in_channel, 1),
            nn.Sigmoid()
        )
        self.spatial_conv1 = nn.Conv2d(4, 1, kernel_size=7, padding=3)
        self.spatial_conv2 = nn.Conv2d(4, 1, kernel_size=7, padding=3)
        self.catconv = dsconv_3x3(in_channel * 2, in_channel)
        self.softmax = nn.Softmax(0)

    def forward(self, t1, t2, log=False, module_name=None, img_name=None):
        # 探究前后向对模型的影响
        diff = torch.abs(t1 - t2)
        diff_w = self.diffw(diff)
        # channel part
        t1_channel_avg_pool = self.avg_pool(t1)  # b,c,1,1
        t1_channel_max_pool = self.max_pool(t1)  # b,c,1,1
        t2_channel_avg_pool = self.avg_pool(t2)  # b,c,1,1
        t2_channel_max_pool = self.max_pool(t2)  # b,c,1,1

        channel_pool = torch.cat([t1_channel_avg_pool, t1_channel_max_pool,
                                  t2_channel_avg_pool, t2_channel_max_pool],
                                 dim=2).squeeze(-1).transpose(1, 2)  # b,4,c
        t1_channel_attention = self.channel_conv1(channel_pool)  # b,1,c
        t2_channel_attention = self.channel_conv2(channel_pool)  # b,1,c
        # 堆叠两个张量在批次维度为一个
        channel_stack = torch.stack([t1_channel_attention, t2_channel_attention],
                                    dim=0)  # 2,b,1,c
        # input＝3维数据，则默认维度为0, 如果input＝４维数据，则默认维度为1 通道维度映射0-1 和为1
        channel_stack = self.softmax(channel_stack).transpose(-1, -2).unsqueeze(-1)  # 2,b,c,1,1

        # spatial part
        t1_spatial_avg_pool = torch.mean(t1, dim=1, keepdim=True)  # b,1,h,w
        t1_spatial_max_pool = torch.max(t1, dim=1, keepdim=True)[0]  # b,1,h,w
        t2_spatial_avg_pool = torch.mean(t2, dim=1, keepdim=True)  # b,1,h,w
        t2_spatial_max_pool = torch.max(t2, dim=1, keepdim=True)[0]  # b,1,h,w
        spatial_pool = torch.cat([t1_spatial_avg_pool, t1_spatial_max_pool,
                                  t2_spatial_avg_pool, t2_spatial_max_pool], dim=1)  # b,4,h,w
        t1_spatial_attention = self.spatial_conv1(spatial_pool)  # b,1,h,w
        t2_spatial_attention = self.spatial_conv2(spatial_pool)  # b,1,h,w
        spatial_stack = torch.stack([t1_spatial_attention, t2_spatial_attention], dim=0)  # 2,b,1,h,w
        spatial_stack = self.softmax(spatial_stack)  # 2,b,1,h,w

        # fusion part, add 1 means residual add
        # 如果某一位置的注意力权重较小，那么加上常数1后可以保证至少有一定比例的原始特征被保留下来。这有助于避免特征信息的丢失，从而有助于模型的训练和优化。
        stack_attention = channel_stack + spatial_stack + 1  # 2,b,c,h,w
        fuse1 = stack_attention[0] * t1 * diff_w
        fuse2 = stack_attention[1] * t2 * diff_w  # b,c,h,w
        fuse = self.catconv(torch.cat([fuse1, fuse2], dim=1))

        return fuse


###################################################################
# ########################## NETWORK ##############################
###################################################################
class AACRNet(nn.Module):
    def __init__(self):
        super(AACRNet, self).__init__()
        # params
        channel_list = [32, 64, 128, 256, 512]
        # backbone
        self.backbone = ResNet()
        self.atf1 = TDFM(in_channel=channel_list[1])
        self.atf2 = TDFM(in_channel=channel_list[2])
        self.atf3 = TDFM(in_channel=channel_list[3])
        self.atf4 = TDFM(in_channel=channel_list[4])
        self.sematic = dsconv_3x3(64, 512)
        # positioning
        self.mask_gen = SAM(64)
        # focus
        self.focus4 = Focus(512, 512)
        self.focus3 = Focus(256, 512)
        self.focus2 = Focus(128, 256)
        self.focus1 = Focus(64, 128)

    def forward(self, t1, t2, log=False, img_name=None):
        t1_2, t1_3, t1_4, t1_5 = self.backbone(t1)
        t2_2, t2_3, t2_4, t2_5 = self.backbone(t2)

        d1 = self.atf1(t1_2, t2_2)
        d2 = self.atf2(t1_3, t2_3)
        d3 = self.atf3(t1_4, t2_4)
        d4 = self.atf4(t1_5, t2_5)

        sematic, coarse = self.mask_gen(d4, d3, d2)
        sematic = self.sematic(sematic)
        # focus
        focus4, predict4 = self.focus4(d4, sematic, coarse)
        focus3, predict3 = self.focus3(d3, focus4, predict4)
        focus2, predict2 = self.focus2(d2, focus3, predict3)
        focus1, predict1 = self.focus1(d1, focus2, predict2)

        # rescale
        coarse = F.interpolate(coarse, size=t1.size()[2:], mode='bilinear', align_corners=True)
        predict4 = F.interpolate(predict4, size=t1.size()[2:], mode='bilinear', align_corners=True)
        predict3 = F.interpolate(predict3, size=t1.size()[2:], mode='bilinear', align_corners=True)
        predict2 = F.interpolate(predict2, size=t1.size()[2:], mode='bilinear', align_corners=True)
        predict1 = F.interpolate(predict1, size=t1.size()[2:], mode='bilinear', align_corners=True)
        return predict1, predict2, predict3, predict4, coarse


if __name__ == '__main__':
    x1 = torch.randn(3, 3, 256, 256)
    x2 = torch.randn(3, 3, 256, 256)
    model = AACRNet()
    res = model(x1, x2)
    # 计算模型参数量
    flops, params = profile(model, inputs=(x1, x2))
    flops, params = clever_format([flops, params], "%.3f")
    print(params)
    print(flops)
