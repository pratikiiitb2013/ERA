import torch.nn as nn
import torch.nn.functional as F
def norms(normtype, embedding):
  if normtype=='bn':
     return nn.BatchNorm2d(embedding)
  elif normtype=='ln':
     return nn.GroupNorm(1, embedding)
  else:
    return nn.GroupNorm(4, embedding)

def custom_layer(in_c, out_c, normtype, pool):
        layers = [
            nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=(3, 3), padding=1, bias=False),
        ]
        if pool:
            layers.append(
                nn.MaxPool2d(2, 2),
            )
        layers.append(norms(normtype, out_c))
        layers.append(nn.ReLU())
        block = nn.Sequential(*layers)
        return block

class Net(nn.Module):
    def __init__(self, normtype):
        super(Net, self).__init__()
        #prep layer
        self.preplayer = custom_layer(3, 64, 'bn', False) # output_size = 32, RF 3
        self.layer1_X = custom_layer(64, 128, 'bn', True)
        self.layer1_R = nn.Sequential(
            custom_layer(128, 128, 'bn', False),
            custom_layer(128, 128, 'bn', False)
        )
        self.layer2_X = custom_layer(128, 256, 'bn', True)
        self.layer3_X = custom_layer(256, 512, 'bn', True)
        self.layer3_R = nn.Sequential(
            custom_layer(512, 512, 'bn', False),
            custom_layer(512, 512, 'bn', False)
        )
        self.pool = nn.MaxPool2d(4, 4)
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = self.preplayer(x)
        x1 = self.layer1_X(x)
        x = self.layer1_R(x1) + x1
        x = self.layer2_X(x)
        x1 = self.layer3_X(x)
        x = self.layer3_R(x1) + x1
        x = self.pool(x)
        x = x.view(-1, 512)
        x = self.fc(x)
        return nn.Softmax(dim=1)(x)