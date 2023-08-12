import torch.nn as nn
import torch.nn.functional as F

def norms(normtype, embedding):
  if normtype=='bn':
     return nn.BatchNorm2d(embedding)
  elif normtype=='ln':
     return nn.GroupNorm(1, embedding)
  else:
    return nn.GroupNorm(4, embedding)

class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=3, padding=1, groups=nin)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class Net(nn.Module):
  def __init__(self, normtype):
    super(Net, self).__init__()
    self.prepLayer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            norms(normtype, 64),
            nn.ReLU(),
        )
    ###########################################################
    self.L1_conv = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            norms(normtype, 128),
            nn.ReLU(),
        )
    self.L1_residual = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            norms(normtype, 128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            norms(normtype, 128),
            nn.ReLU(),
        )
    ###########################################################
    self.L2_conv = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            norms(normtype, 256),
            nn.ReLU(),
        )
    ###########################################################
    self.L3_conv = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            norms(normtype, 512),
            nn.ReLU(),
        )
    self.L3_residual = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, bias=False),
            norms(normtype, 512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, bias=False),
            norms(normtype, 512),
            nn.ReLU(),
        )
    ###########################################################
    self.pool = nn.MaxPool2d(4, 4)
    self.fc_with_1X1 = nn.Conv2d(in_channels=512, out_channels=10, kernel_size=(1, 1), bias=False)




  def forward(self, x):
    x = self.prepLayer(x)

    x1 = self.L1_conv(x)
    r1 = self.L1_residual(x1)
    x = x1 + r1

    x = self.L2_conv(x)

    x3 = self.L3_conv(x)
    r3 = self.L3_residual(x3)
    x = x3 + r3

    x = self.pool(x)

    x = self.fc_with_1X1(x)

    x = x.view(-1, 10)

    return x

