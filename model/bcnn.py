import torch
import torch.nn as nn
from torchvision.models import resnet18


class ResBCNN(nn.Module):
    def __init__(self):
        super(ResBCNN, self).__init__()
        self.features = nn.Sequential(
            resnet18().conv1,
            resnet18().relu,
            resnet18().maxpool,
            resnet18().layer1,
            resnet18().layer2,
            resnet18().layer3,
            resnet18().layer4
        )
        self.classifiers = nn.Sequential(nn.Linear(512**2, 14))

    def forward(self, x):
        x = self.features(x)  # b*512*8*8
        batch_size = x.size(0)
        x = x.view(batch_size, 512, x.size(2)**2)  # b * 512 * 64
        x = (torch.bmm(x, torch.transpose(x, 1, 2)) / (x.size(2) ** 2)).view(batch_size, -1)  # 512*62 X 62*512 => 512**2 => b*262144
        x = torch.nn.functional.normalize(torch.sign(x) * torch.sqrt(torch.abs(x) + 1e-10))  # Improved BCNN
        x = self.classifiers(x)
        return x


if __name__ == "__main__":
    model = ResBCNN()
    x = torch.randn((2, 3, 256, 256), dtype=torch.float32)
    y = model(x)
    print(f'input: {x.shape}')
    print(f'output: {y.shape}')
    from torchsummary import summary
    summary_vision = summary(model, (3, 256, 256))