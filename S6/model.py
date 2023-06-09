import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import GetCorrectPredCount

from tqdm import tqdm

import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3) #input 28x28x3 OUtput 26x26x16 RF 3
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 16, 3) #input 26x26x16 OUtput 24x24x16 RF 5
        self.bn2 = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(16, 16, 3) #input 24x24x16 OUtput 22x22x16 RF 7
        self.bn3 = nn.BatchNorm2d(16)
        
        self.pool1 = nn.MaxPool2d(2, 2)  #input 22x22x16 OUtput 11x11x16 RF 8
  
     

        self.conv4 = nn.Conv2d(16, 16, 1) #input 11x11x16 OUtput 11x11x8 RF 8
        self.bn4 = nn.BatchNorm2d(16)
        
        self.conv5 = nn.Conv2d(16, 16, 3) #input 11x11x8 OUtput 9x9x16 RF 12
        self.bn5 = nn.BatchNorm2d(16)
        self.dp5 = nn.Dropout2d(0.1)

        self.conv6 = nn.Conv2d(16, 16, 3) #input 9x9x16 OUtput 7x7x16 RF 16
        self.bn6 = nn.BatchNorm2d(16)
        self.dp6 = nn.Dropout2d(0.1)

        self.conv7 = nn.Conv2d(16, 16, 3) #input 7x7x16 OUtput 5x5x16 RF 20
        self.bn7 = nn.BatchNorm2d(16)
        self.dp7 = nn.Dropout2d(0.1)

        self.conv8 = nn.Conv2d(16, 16, 3, padding=1) #input 5x5x16 OUtput 1x1x10 RF 2
        self.bn8 = nn.BatchNorm2d(16)
        self.dp8 = nn.Dropout2d(0.2)

        self.conv9 = nn.Conv2d(16, 16, 3, padding=1) #input 5x5x16 OUtput 1x1x10 RF 2
        self.bn9 = nn.BatchNorm2d(16)
        self.dp9 = nn.Dropout2d(0.2)
        
        self.gap = nn.AvgPool2d(kernel_size=5)
        self.linear = nn.Linear(16,10)

    def forward(self, x):
        x = (self.bn1(F.relu(self.conv1(x)))) 
        x = (self.bn2(F.relu(self.conv2(x)))) 
        x = self.pool1((self.bn3(F.relu(self.conv3(x))))) 
        x = self.bn4(F.relu(self.conv4(x))) 
        x = self.dp5(self.bn5(F.relu(self.conv5(x)))) 
        x = self.dp6(self.bn6(F.relu(self.conv6(x)))) 
        x = self.dp7(self.bn7(F.relu(self.conv7(x)))) 
        x = self.dp8(self.bn8(F.relu(self.conv8(x)))) 
        x = self.dp9(self.bn9(F.relu(self.conv9(x)))) 
        x = self.gap(x)
        #x = self.conv8(x) 
        x = x.view(-1, 16)
        x = self.linear(x)
        return F.log_softmax(x)
    

def train(model, device, train_loader, optimizer, criterion, train_acc, train_losses):
    
    model.train()
    pbar = tqdm(train_loader)
    
    train_loss = 0
    correct = 0
    processed = 0

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # Predict
        pred = model(data)

        # Calculate loss
        loss = criterion(pred, target)
        train_loss+=loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()
        
        correct += GetCorrectPredCount(pred, target)
        processed += len(data)

        pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

    train_acc.append(100*correct/processed)
    train_losses.append(train_loss/len(train_loader))
    return train_acc, train_losses

def test(model, device, test_loader, criterion, test_acc, test_losses):

    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss

            correct += GetCorrectPredCount(output, target)


    test_loss /= len(test_loader.dataset)
    test_acc.append(100. * correct / len(test_loader.dataset))
    test_losses.append(test_loss)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    return test_acc, test_losses
