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
    self.convblock0 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            norms(normtype, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
    ###########################################################
    self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            norms(normtype, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
    self.convblock1_d = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), dilation=2, padding=2, bias=False),
            norms(normtype, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
    self.convblock2_depth = nn.Sequential(
            depthwise_separable_conv(32, 128),
            norms(normtype, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
    self.conv1X1_and_dilated3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(1, 1), bias=False),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), dilation=8, bias=False),
            # norms(normtype, 32),
            # nn.ReLU(),
        )
    self.skip_with_stride2_1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 1), padding=0, stride=2, bias=False),
            # norms(normtype, 32),
            # nn.ReLU(),
        )
    ############################################################
    self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            norms(normtype, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
    self.convblock4_d = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), dilation=2, padding=2, bias=False),
            norms(normtype, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
    self.convblock5_depth = nn.Sequential(
            depthwise_separable_conv(64, 128),
            norms(normtype, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
    self.conv1X1_and_dilated6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(1, 1), bias=False),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), dilation=4, bias=False),
            # norms(normtype, 32),
            # nn.ReLU(),
        )
    self.skip_with_stride2_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 1), padding=0, stride=2, bias=False),
            # norms(normtype, 32),
            # nn.ReLU(),
        )
    ############################################################
    self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            norms(normtype, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
    self.convblock7_d = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), dilation=2, padding=2, bias=False),
            norms(normtype, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
    self.convblock8_depth = nn.Sequential(
            depthwise_separable_conv(64, 128),
            norms(normtype, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
    self.conv1X1_and_dilated9 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(1, 1), bias=False),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), dilation=2, bias=False),
            # norms(normtype, 32),
            # nn.ReLU(),
        )
    self.skip_with_stride2_3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 1), padding=0, stride=2, bias=False),
            # norms(normtype, 32),
            # nn.ReLU(),
        )
    ############################################################
    self.convblock10 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            norms(normtype, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
    self.convblock10_d = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), dilation=2, padding=2, bias=False),
            norms(normtype, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
    self.convblock11_depth = nn.Sequential(
            depthwise_separable_conv(64, 128),
            norms(normtype, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
    
    self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=4)
        )

    self.convblock12_1X1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        )
    # self.convblock13_1X1 = nn.Sequential(
    #         nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 1), padding=0, bias=False),
    #     )
    # self.convblock14_1X1 = nn.Sequential(
    #         nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
    #     )
    
  def forward(self, x):
    xi = self.convblock0(x)
    # xi = self.skip_with_stride2_1(xi)

    x = self.convblock1(xi) + self.convblock1_d(xi)
    x = self.convblock2_depth(x)
    xi = self.conv1X1_and_dilated3(x) + self.skip_with_stride2_1(xi)


    x = self.convblock4(xi) + self.convblock4_d(xi)
    x = self.convblock5_depth(x)
    xi = self.conv1X1_and_dilated6(x) + self.skip_with_stride2_2(xi)

    x = self.convblock7(xi) + self.convblock7_d(xi)
    x = self.convblock8_depth(x)
    xi = self.conv1X1_and_dilated9(x) + self.skip_with_stride2_3(xi)

    x = self.convblock10(xi) + self.convblock10_d(xi)
    x = self.convblock11_depth(x)

    x = self.gap(x)
    x = self.convblock12_1X1(x)
    # x = self.convblock13_1X1(x)
    # x = self.convblock14_1X1(x)

    x = x.view(-1, 10) #1x1x10> 10
    return F.log_softmax(x, dim=-1)

