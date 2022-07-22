
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self,
        num_classes, dropout=0.5
    ) -> None:
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 196, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(196, 384, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.head = nn.Sequential(
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, num_classes)
        )

    def feature_forward(self, x):
        x = self.feature(x)
        return x

    def head_forward(self, x):
        B = x.size(0)
        x = self.avgpool(x)
        x = x.view(B, -1)
        x = self.head(x)
        return x

    def forward(self, x):
        x = self.feature_forward(x)
        x = self.head_forward(x)
        return x
