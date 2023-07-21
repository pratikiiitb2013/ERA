import torch
from tqdm import tqdm

# train_losses = []
# train_acc = []


def GetCorrectPredCount(pPrediction, pLabels):
  return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

def train(model, device, trainloader, optimizer, criterion, train_acc, train_losses):
    model.train()
    pbar = tqdm(trainloader)
    train_loss = 0
    correct = 0
    processed = 0
   
    for batch_idx, (data, target) in enumerate(pbar):
        # get the inputs
        
        data, target = data.to(device), target.to(device)

        # Init
        optimizer.zero_grad()
        # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes.
        # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

        # forward + backward + optimize
        pred = model(data)
        # Calculate loss
        loss = criterion(pred, target)

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Update pbar-tqdm
        # pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        # correct += pred.eq(target.view_as(pred)).sum().item()

        correct += GetCorrectPredCount(pred, target)

        processed += len(data)

        # pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
        pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
        train_acc.append(100*correct/processed)
        train_acc = 100*correct/processed
        train_losses.append(train_loss/len(trainloader))
        return train_acc, train_losses

