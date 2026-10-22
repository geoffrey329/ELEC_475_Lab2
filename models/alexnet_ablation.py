"""
ELEC 475: Lab 2 - AlexNet-64 Extended Ablation Variants
Each variant makes exactly ONE change to AlexNet-64 + BatchNorm (models/alexnet.py)
so its effect can be measured against that baseline:

    'no_dropout' : remove both Dropout(p=0.5) layers from the classifier
    'half_width' : halve every conv layer's channel count (64,192,384,384,256 -> 32,96,192,192,128)
    'shallow'    : remove conv stage 4 (5 conv layers -> 4)
    'avgpool'    : replace the three overlapping MaxPool(3, s=2) layers with AvgPool(3, s=2)

All variants keep the same 64x64 input, spatial dimensions, and BatchNorm placement.
"""
import torch
import torch.nn as nn

VARIANTS = ["no_dropout", "half_width", "shallow", "avgpool"]


class AlexNet64Ablation(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 3, variant: str = "no_dropout"):
        super(AlexNet64Ablation, self).__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant '{variant}'. Choose from {VARIANTS}.")
        self.variant = variant

        # Channel widths for conv stages 1-5
        c1, c2, c3, c4, c5 = (64, 192, 384, 384, 256)
        if variant == "half_width":
            c1, c2, c3, c4, c5 = (32, 96, 192, 192, 128)

        def pool() -> nn.Module:
            if variant == "avgpool":
                return nn.AvgPool2d(kernel_size=3, stride=2)
            return nn.MaxPool2d(kernel_size=3, stride=2)

        def conv_bn_relu(c_in: int, c_out: int, k: int, s: int, p: int) -> list:
            return [nn.Conv2d(c_in, c_out, kernel_size=k, stride=s, padding=p),
                    nn.BatchNorm2d(c_out),
                    nn.ReLU(inplace=True)]

        layers = []
        layers += conv_bn_relu(in_channels, c1, 5, 2, 2) + [pool()]   # Stage 1: (c1, 32, 32) -> (c1, 15, 15)
        layers += conv_bn_relu(c1, c2, 5, 1, 2) + [pool()]            # Stage 2: (c2, 15, 15) -> (c2, 7, 7)
        layers += conv_bn_relu(c2, c3, 3, 1, 1)                       # Stage 3: (c3, 7, 7)
        if variant == "shallow":
            layers += conv_bn_relu(c3, c5, 3, 1, 1) + [pool()]        # Stage 5 directly after stage 3
        else:
            layers += conv_bn_relu(c3, c4, 3, 1, 1)                   # Stage 4: (c4, 7, 7)
            layers += conv_bn_relu(c4, c5, 3, 1, 1) + [pool()]        # Stage 5: (c5, 7, 7) -> (c5, 3, 3)
        self.features = nn.Sequential(*layers)

        p_drop = 0.0 if variant == "no_dropout" else 0.5
        self.classifier = nn.Sequential(
            nn.Dropout(p=p_drop),
            nn.Linear(c5 * 3 * 3, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p_drop),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)


if __name__ == "__main__":
    from torchinfo import summary
    for v in VARIANTS:
        model = AlexNet64Ablation(num_classes=10, in_channels=3, variant=v)
        out = model(torch.randn(1, 3, 64, 64))
        params = sum(p.numel() for p in model.parameters())
        print(f"{v:<12} output {tuple(out.shape)}  params {params:,}")
    summary(AlexNet64Ablation(variant="half_width"), input_size=(1, 3, 64, 64))
