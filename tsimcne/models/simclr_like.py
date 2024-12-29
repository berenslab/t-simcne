import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def make_model(
    backbone,
    projection_head,
    seed=None,
    in_channel=3,
    hidden_dim=1024,
    out_dim=128,
    **kwargs,
):
    if seed is not None:
        torch.manual_seed(seed)

    if isinstance(backbone, str) and isinstance(projection_head, str):
        return ResNetFC(
            backbone=backbone,
            projection_head=projection_head,
            in_channel=in_channel,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            **kwargs,
        )

    elif isinstance(backbone, str):
        try:
            model_func, backbone_dim = model_dict[backbone]
        except KeyError:
            raise ValueError(
                f"{backbone = !r} not registered in model_dict."
                f"  Available backbones are {model_dict.keys()}"
            )
        backbone = model_func(in_channel=in_channel)
        # projection_head =

        return ContrastiveFC(
            backbone=backbone,
            projection_head=projection_head,
            backbone_dim=backbone_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
        )

    elif isinstance(projection_head, str):
        if "backbone_dim" not in kwargs or kwargs["backbone_dim"] is None:
            raise ValueError(
                "'backbone_dim' has not been"
                " specified and cannot be determined"
                " when passing custom backbone."
            )
        projection_head = make_projection_head(
            projection_head,
            in_dim=kwargs["backbone_dim"],
            hidden_dim=hidden_dim,
            out_dim=out_dim,
        )
        return ContrastiveFC(
            backbone=backbone,
            projection_head=projection_head,
            backbone_dim=kwargs["backbone_dim"],
            hidden_dim=hidden_dim,
            out_dim=out_dim,
        )
    else:
        return ContrastiveFC(
            backbone=backbone,
            projection_head=projection_head,
            backbone_dim=None,
            hidden_dim=None,
            out_dim=out_dim,
        )


def make_projection_head(name="mlp", in_dim=512, hidden_dim=1024, out_dim=128):
    return FCNetwork(
        in_dim=in_dim, feat_dim=out_dim, hidden_dim=hidden_dim, arch=name
    )


class ContrastiveFC(nn.Module):
    def __init__(
        self,
        backbone,
        projection_head,
        backbone_dim,
        hidden_dim,
        out_dim=128,
        in_channel=3,
    ):
        super(ContrastiveFC, self).__init__()
        self.backbone = backbone
        self.projection_head = projection_head
        self.backbone_dim = backbone_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.in_channel = in_channel

    def forward(self, x):
        h = self.backbone(x)
        z = self.projection_head(h)
        return z, h


class ResNetFC(nn.Module):
    def __init__(
        self,
        backbone="resnet18_simclr",
        projection_head="mlp",
        in_channel=3,
        out_dim=128,
        hidden_dim=1024,
        backbone_dim=None,
    ):
        super(ResNetFC, self).__init__()
        try:
            model_func, _backbone_dim = model_dict[backbone]
        except KeyError:
            raise ValueError(
                f"{backbone=!r} not registered in model_dict."
                f"  Available backbones are {model_dict.keys()}"
            )
        if backbone_dim is not None:
            if backbone_dim != _backbone_dim:
                raise ValueError(
                    f"Expected passed {backbone_dim=!r} to match the "
                    f"one found in the model dict, but got {_backbone_dim}."
                )
        self.backbone_dim = _backbone_dim
        self.out_dim = out_dim
        self.backbone = model_func(in_channel=in_channel)

        self.projection_head = make_projection_head(
            projection_head,
            in_dim=self.backbone_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
        )

    def forward(self, x):
        h = self.backbone(x)
        z = self.projection_head(h)
        return z, h


class FCNetwork(nn.Module):
    "Fully-connected network"

    def __init__(self, in_dim=784, feat_dim=128, hidden_dim=100, arch="mlp"):
        super(FCNetwork, self).__init__()

        self.in_dim = in_dim
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim

        self.flatten = nn.Flatten()
        if arch == "mlp":
            self.layers = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, feat_dim),
            )
        elif arch == "linear":
            self.layers = nn.Sequential(nn.Linear(in_dim, feat_dim))
        else:
            raise ValueError(f"Unknown network {arch = !r}")

    def forward(self, x):
        x = self.flatten(x)
        logits = self.layers(x)
        return logits


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out, inplace=True)
        if self.is_last:
            return out, preact
        else:
            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out, inplace=True)
        if self.is_last:
            return out, preact
        else:
            return out


class ResNet(nn.Module):
    def __init__(
        self, block, num_blocks, in_channel=3, zero_init_residual=False
    ):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(
            in_channel, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch, so that
        # the residual branch starts with zeros, and each residual
        # block behaves like an identity. This improves the model by
        # 0.2~0.3% according to: https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, layer=100):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out


class AdaptedConvNet(nn.Module):
    """Creates a torchvision model with a modified stride.

    This class takes a ``modelfunc`` as a parameter which should
    instantiate a torchvision model as in
    https://pytorch.org/vision/main/models.html which will then have
    its stride modified so that it's closer to the adapted resnet18
    (as described in the SimCLR paper).

    """

    def __init__(self, modelfunc, in_channel=3, **kwargs):
        super(AdaptedConvNet, self).__init__()
        self.in_channel = in_channel
        self.model = modelfunc(**kwargs)
        conv1 = self.model.features[0][0]
        conv1.stride = (1, 1)

    def forward(self, x):
        feat = self.model.features(x)
        return self.model.avgpool(feat)


def pytorch_resnet(in_channel=3):
    model = torchvision.models.resnet18()
    model.fc = torch.nn.Identity()
    return model


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


def mobilenet_v3_large(**kwargs):
    return AdaptedConvNet(torchvision.models.mobilenet_v3_large, **kwargs)


def mobilenet_v3_small(**kwargs):
    return AdaptedConvNet(torchvision.models.mobilenet_v3_small, **kwargs)


def efficientnet_v2_s(**kwargs):
    return AdaptedConvNet(torchvision.models.efficientnet_v2_s, **kwargs)


def efficientnet_v2_m(**kwargs):
    return AdaptedConvNet(torchvision.models.efficientnet_v2_m, **kwargs)


def efficientnet_v2_l(**kwargs):
    return AdaptedConvNet(torchvision.models.efficientnet_v2_l, **kwargs)


model_dict = {
    "resnet18": [pytorch_resnet, 512],
    "resnet18_sm_kernel": [resnet18, 512],
    "resnet34": [resnet34, 512],
    "resnet50": [resnet50, 2048],
    "resnet101": [resnet101, 2048],
    "mobilenetv3_small": [mobilenet_v3_small, 576],
    "mobilenetv3_large": [mobilenet_v3_large, 960],
    "efficientnet_v2_s": [efficientnet_v2_s, 1280],
    "efficientnet_v2_m": [efficientnet_v2_m, 1280],
    "efficientnet_v2_l": [efficientnet_v2_l, 1280],
}
