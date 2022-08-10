import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import ProjectBase


def make_model(**kwargs):
    return ResNetFC(**kwargs)


def mutate_model(
    model: torch.nn.Module,
    change: str = "nothing",
    freeze: bool = None,
    proj_head="mlp",
    out_dim: int = 2,
    hidden_dim=None,
    # **kwargs,
):
    """mutate a given model for further finetuning.

    This function does two things:
        1. It changes whether the model requires a gradient (via `freeze`).
        2. It swaps out a part of the model (currently only in the proj. head).
           This is controlled via the parameter `change`.

    By default the function does not do anything and will just return the model
    as is.
    """
    if freeze is not None:
        model.requires_grad_(freeze)

    if change == "lastlin":
        # swap out the last linear layer of the projection head
        last_layer = model.projection_head.layers[-1]
        dim = last_layer.weight.size(1)
        model.projection_head.layers[-1] = nn.Linear(dim, out_dim)

    elif change == "proj_head":
        # swap out the entire projection head
        in_dim = model.backbone_dim
        if hidden_dim is None:
            # try to infer the dim from the previous projection head
            hidden_dim = model.projection_head.layers[0].weight.size(0)
        model.projection_head = make_projection_head(
            proj_head, in_dim, hidden_dim, out_dim
        )

    elif change == "nothing":
        pass

    else:
        raise ValueError(
            f"Requested to {change = !r}, but I don't know what to do"
        )

    return model


def make_projection_head(name="mlp", in_dim=512, hidden_dim=1024, out_dim=128):
    if name == "mlp":
        projection_head = FCNetwork(
            in_dim=in_dim, feat_dim=out_dim, hidden_dim=hidden_dim
        )
    else:
        raise ValueError(f"Unknown projection head {name = !r}")

    return projection_head


class SimCLRModel(ProjectBase):
    def __init__(self, path, random_state=None, **kwargs):
        super().__init__(path, random_state=random_state)
        self.kwargs = kwargs

    def get_deps(self):
        return []

    def load(self):
        pass

    def compute(self):
        self.model = make_model(**self.kwargs)

    def save(self):
        save_data = dict(model=self.model, model_sd=self.model.state_dict())
        self.save_lambda_alt(self.outdir / "model.pt", save_data, torch.save)


class FinetuneSimCLRModel(SimCLRModel):
    def get_deps(self):
        supdeps = super().get_deps()
        return supdeps + [self.indir / "model.pt"]

    def load(self):
        self.state_dict = torch.load(self.indir / "model.pt")
        self.model = self.state_dict["model"]

    def compute(self):
        self.model = mutate_model(self.model, **self.kwargs)

    def save(self):
        # remove old values that might be present
        self.state_dict.pop("model", None)
        self.state_dict.pop("model_sd", None)

        save_data = dict(
            **self.state_dict,
            model=self.model,
            model_sd=self.model.state_dict(),
        )
        self.save_lambda_alt(self.outdir / "model.pt", save_data, torch.save)


class ResNetFC(nn.Module):
    def __init__(
        self,
        backbone="resnet18",
        proj_head="mlp",
        in_channel=3,
        out_dim=128,
        hidden_dim=1024,
    ):
        super(ResNetFC, self).__init__()
        try:
            model_func, backbone_dim = model_dict[backbone]
        except KeyError:
            raise ValueError(
                f"{backbone = !r} not registered in model_dict."
                f"  Available backbones are {model_dict.keys()}"
            )
        self.backbone_dim = backbone_dim
        self.backbone = model_func(in_channel=in_channel)

        self.projection_head = make_projection_head(
            proj_head,
            in_dim=backbone_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
        )

    def forward(self, x):
        h = self.backbone(x)
        z = self.projection_head(h)
        return z, h


class FCNetwork(nn.Module):
    "Fully-connected network"

    def __init__(self, in_dim=784, feat_dim=128, hidden_dim=100):
        super(FCNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feat_dim),
        )

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


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


model_dict = {
    "resnet18": [resnet18, 512],
    "resnet34": [resnet34, 512],
    "resnet50": [resnet50, 2048],
    "resnet101": [resnet101, 2048],
}