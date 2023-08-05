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
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            norms(normtype, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
        ) # output_size = 32, RF 3
        self.convblock1_d = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), dilation=2, padding=2, bias=False),
            norms(normtype, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
        ) 
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            norms(normtype, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
        ) # output_size = 32, RF 5
        self.convblock2_d = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=96, kernel_size=(3, 3), dilation=2, padding=2, bias=False),
            norms(normtype, 96),
            nn.ReLU(),
            nn.Dropout(0.1),
        ) 
        self.convblock3 = nn.Sequential(
            depthwise_separable_conv(32, 96),
            norms(normtype, 96),
            nn.ReLU(),
            nn.Dropout(0.1),
        ) # output_size = 32, RF 7
        self.convblock3_1x1 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=16, kernel_size=(1, 1), bias=False)
        ) # output_size = 32, RF 9
        self.dilated3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), dilation=8, bias=False),
            norms(normtype, 16),
            nn.ReLU(),
        ) 
        self.jump3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), dilation=8, bias=False),
            norms(normtype, 16),
            nn.ReLU(),
        )# output_size = 32, RF 5
         # output_size = 16, RF 10
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            norms(normtype, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
        ) # output_size = 16, RF 18
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            norms(normtype, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
        ) # output_size = 16, RF 18
        self.convblock6 = nn.Sequential(
            depthwise_separable_conv(64, 112),
            norms(normtype, 112),
            nn.ReLU(),
            nn.Dropout(0.1),
        ) # output_size = 16, RF 26
        self.convblock6_1x1 = nn.Sequential(
            nn.Conv2d(in_channels=112, out_channels=32, kernel_size=(1, 1), bias=False)
        ) # output_size = 32, RF 9
        self.dilated6 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), dilation=4, bias=False),
            norms(normtype, 32),
            nn.ReLU(),
        ) 
        self.jump6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), dilation=4, bias=False),
            norms(normtype, 32),
            nn.ReLU(),
        ) # output_size = 16, RF 26
        # output_size = 8, RF 32
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            norms(normtype, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
        ) # output_size = 16, RF 26
        self.convblock7_d = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), dilation=2, padding=2, bias=False),
            norms(normtype, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            norms(normtype, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
        ) # output_size = 16, RF 26
        
        self.convblock9 = nn.Sequential(
            depthwise_separable_conv(64,128),
            norms(normtype, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
        ) # output_size = 16, RF 26
        self.convblock9_1x1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(1, 1), bias=False)
        ) # output_size = 32, RF 9
        self.strided9 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, stride=2, bias=False),
            norms(normtype, 32),
            nn.ReLU(),
        ) 
        self.jump9 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, stride=2, bias=False),
            norms(normtype, 32),
            nn.ReLU(),
        )
        self.convblock10 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            norms(normtype, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
        ) # output_size = 16, RF 26
        self.convblock11 = nn.Sequential(
            depthwise_separable_conv(64,96),
            norms(normtype, 96),
            nn.ReLU(),
            nn.Dropout(0.1),
        ) # output_size = 16, RF 26
        self.convblock12 = nn.Sequential(
            depthwise_separable_conv(96,128),
            norms(normtype, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
        ) # output_size = 16, RF 26
        self.convblock12_1x1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(1, 1), bias=False)
        ) # output_size = 32, RF 9
        
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=4)
        ) # output_size = 1
        self.fc1 = nn.Linear(32, 10)
        #self.fc2 = nn.Linear(64, 10)

    def forward(self, xi):
        x1 = self.convblock1(xi) #3
        x2 = self.convblock1_d(xi) #5
        x = self.convblock2(x1) + x2 #5 #adding dilated conv
        x2 = self.convblock2_d(x1) #5
        x = self.convblock3(x) + x2 #5 #adding dilated conv
        x = self.convblock3_1x1(x) #9
        xi = self.dilated3(x) + self.jump3(xi) #10  #Maxpool but using dilation
        x1 = self.convblock4(xi) #18
        x = self.convblock5(x1)
        x = self.convblock6(x) 
        x = self.convblock6_1x1(x)
        xi = self.dilated6(x) + self.jump6(xi) #28 #Maxpool but using dilation
        x1 = self.convblock7(xi) #52
        x2 = self.convblock7_d(xi) #5
        x = self.convblock8(x1) + x2 #44 #adding dilated conv
        x = self.convblock9(x) #44 #adding dilated conv
        x = self.convblock9_1x1(x)
        xi = self.strided9(x) + self.jump9(xi) #28 #Maxpool but using striding
        x1 = self.convblock10(xi) #52
        x = self.convblock11(x1) #44
        x = self.convblock12(x) #44
        x = self.convblock12_1x1(x) + xi
        x = self.gap(x)
        x = x.view(-1, 32)
        x = (self.fc1(x))
        return x