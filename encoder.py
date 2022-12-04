import torch.nn as nn

class Net(nn.Module):
    def __init__(self, num_classes=128):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(2816, 256),
            # nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256),
            # nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 128),
        )
    def forward(self, x):
        x = self.net(x)
        return x
