import torch
import torchvision
import os
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
import _pickle as pickle
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import pandas as pd
import torchvision.models as models
import ast
from typing import Union, List, Dict, Any, cast
import torch.optim as optim
import matplotlib.pyplot as plt

class net(nn.Module):

    def __init__(
        self,
        features: nn.Module,
        num_classes: int = 10,
        init_weights: bool = True,
    ) -> None:
        
        super(net, self).__init__()

        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(128 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()  

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers() -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 1

    cfg = [64, 64, 'M', 128, 128, 'M']
    
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def load_data():
    # Load normalized data
    train_data = torchvision.datasets.MNIST('./files/', train=True, download=True,
                                  transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                      (0.1307,), (0.3081,))]))
    train_loader = torch.utils.data.DataLoader(train_data,
                   batch_size=batch_size_train, shuffle=True)

    test_data = torchvision.datasets.MNIST('./files/', train=False, download=True,
                                 transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                     (0.1307,), (0.3081,))
                                 ]))
    test_loader = torch.utils.data.DataLoader(test_data,
                  batch_size=batch_size_test, shuffle=True)

    return train_loader, test_loader

def train(epoch, device):
  network.train()

  for batch_idx, (data, target) in enumerate(train_loader):
    # set the gradients to zero
    optimizer.zero_grad()

    # load data to gpu
    data = data.to(device)
    target = target.to(device)

    # forward pass
    output = network(data)

    # calculate loss
    loss = F.nll_loss(F.log_softmax(output), target)
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
      # train_losses.append(loss.item())
      # train_counter.append(
      #   (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
      torch.save(network.state_dict(), './results/model.pth')
      torch.save(optimizer.state_dict(), './results/optimizer.pth')

  return loss.item()

def test(device):
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      data = data.to(device)
      target = target.to(device)
      output = network(data)
      test_loss += F.nll_loss(F.log_softmax(output), target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  # test_losses.append(test_loss)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))
  return test_loss

if __name__ == '__main__':

    ## Hyperparameters
    n_epochs = 5
    # batch_size_train = 256
    # batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10

    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    train_loader, test_loader = load_data()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('Training network using: ', device)

    network = net(make_layers())
    optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                          momentum=momentum)
    network = network.to(device)

    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = []

    test(device)
    for epoch in range(1, n_epochs + 1):
      train_loss = train(epoch, device)
      test_loss = test(device)

      train_losses.append(train_loss)
      train_counter.append(epoch)
      test_losses.append(test_loss)
      test_counter.append(epoch)

    print(train_losses)
    print(train_counter)
    print(test_losses)
    print(test_counter)
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('Epoch')
    plt.ylabel('negative log likelihood loss')
    fig
    plt.show()

    # with torch.no_grad():
    #   output = network(example_data)

    # fig = plt.figure()
    # for i in range(6):
    #   plt.subplot(2,3,i+1)
    #   plt.tight_layout()
    #   plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    #   plt.title("Prediction: {}".format(
    #     output.data.max(1, keepdim=True)[1][i].item()))
    #   plt.xticks([])
    #   plt.yticks([])
    # fig