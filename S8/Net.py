import torch.nn as nn
import torch.nn.functional as F
def norms(normtype, embedding):
  if normtype=='bn':
     print('####Batch Norm')
     return nn.BatchNorm2d(embedding)
  elif normtype=='ln':
     return nn.GroupNorm(1, embedding)
  else:
    return nn.GroupNorm(4, embedding)

class Net(nn.Module):
    def __init__(self, normtype):
        super(Net, self).__init__()
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            norms(normtype, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
        ) # output_size = 32, RF 3
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            norms(normtype, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
        ) # output_size = 32, RF 5
        self.convblock3_1x1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1, 1), bias=False)
        ) # output_size = 32, RF 9
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
         # output_size = 16, RF 10
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            norms(normtype, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
        ) # output_size = 16, RF 18
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            norms(normtype, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
        ) # output_size = 16, RF 26
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=48, kernel_size=(3, 3), padding=1, bias=False),
            norms(normtype, 48),
            nn.ReLU(),
            nn.Dropout(0.2),
        ) # output_size = 16, RF 26
        self.convblock6_1x1 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=16, kernel_size=(1, 1), bias=False)
        ) # output_size = 32, RF 9
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # output_size = 8, RF 32
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            norms(normtype, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
        ) # output_size = 16, RF 26
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            norms(normtype, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
        ) # output_size = 16, RF 26
        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=48, kernel_size=(3, 3), padding=1, bias=False),
            norms(normtype, 48),
            nn.ReLU(),
            nn.Dropout(0.2),
        ) # output_size = 16, RF 26
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=8)
        ) # output_size = 1
        self.fc1 = nn.Linear(48, 10)
        #self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x1 = self.convblock1(x) #3
        x = self.convblock2(x1) #5
        x = self.convblock3_1x1(x) #9
        x = x + x1
        x = self.pool(x) #10
        x1 = self.convblock3(x) #18
        x = self.convblock4(x) #26
        x = self.convblock5(x) #26
        x = self.convblock6_1x1(x)
        x = x + x1
        x = self.pool2(x) #28
        x = self.convblock7(x) #52
        x = self.convblock8(x) #44
        x = self.convblock9(x) #44
        x = self.gap(x)
        x = x.view(-1, 48)
        x = (self.fc1(x))
        return x