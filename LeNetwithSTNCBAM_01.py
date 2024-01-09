# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 16:39:50 2023

@author: KFS
"""


from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.nn import init
from torch.utils.tensorboard import SummaryWriter
from mpl_toolkits.mplot3d import Axes3D
# import plotly.graph_objects as go

plt.ion()   # interactive mode
######################################################################
# Loading the data
use_cuda=True

from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
# Training dataset
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='./MNIST_data/', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])), batch_size=64, shuffle=True, num_workers=0)
# Test dataset
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='./MNIST_data/', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])), batch_size=64, shuffle=True, num_workers=0)










class ChannelAttention(nn.Module):
    def __init__(self, in_planes=512, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 8, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 8, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

# class ChannelAttention(nn.Module):
#     def __init__(self,channel,reduction=16):
#         super().__init__()
#         self.maxpool=nn.AdaptiveMaxPool2d(1)
#         self.avgpool=nn.AdaptiveAvgPool2d(1)
#         self.se=nn.Sequential(
#             nn.Conv2d(channel,channel//reduction,1,bias=False),
#             nn.ReLU(),
#             nn.Conv2d(channel//reduction,channel,1,bias=False)
#         )
#         self.sigmoid=nn.Sigmoid()
    
#     def forward(self, x) :
#         max_result=self.maxpool(x)
#         avg_result=self.avgpool(x)
#         max_out=self.se(max_result)
#         avg_out=self.se(avg_result)
#         output=self.sigmoid(max_out+avg_out)
#         return output

# class SpatialAttention(nn.Module):
#     def __init__(self,kernel_size=7):
#         super().__init__()
#         self.conv=nn.Conv2d(2,1,kernel_size=kernel_size,padding=kernel_size//2)
#         self.sigmoid=nn.Sigmoid()
    
#     def forward(self, x) :
#         max_result,_=torch.max(x,dim=1,keepdim=True)
#         avg_result=torch.mean(x,dim=1,keepdim=True)
#         result=torch.cat([max_result,avg_result],1)
#         output=self.conv(result)
#         output=self.sigmoid(output)
#         return output

# class CBAMBlock(nn.Module):
#     def __init__(self, channel=512,reduction=16,kernel_size=49):
#         super().__init__()
#         self.ca=ChannelAttention(channel=channel,reduction=reduction)
#         self.sa=SpatialAttention(kernel_size=kernel_size)
#     def init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 init.normal_(m.weight, std=0.001)
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#     def forward(self, x):
#         b, c, _, _ = x.size()
#         residual=x
#         out=x*self.ca(x)
#         out=out*self.sa(out)
#         return out+residual

######################################################################
# Depicting spatial transformer networks
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.ca = ChannelAttention(in_planes=20)
        self.sa = SpatialAttention()

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x
    def stn_2(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x
    
    def stn_3(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x) + self.stn_2(x) + self.stn_3(x)
        
        # Perform the usual forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), 2))      
        
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        
        attention_values = self.ca(x)
        x = attention_values * x
        x = self.sa(x) * x

        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1), attention_values

model = Net().to(device)

######################################################################

def fgsm_attack(input,epsilon,data_grad):
  pert_out = input + epsilon*data_grad.sign()
  pert_out = torch.clamp(pert_out, 0, 1)
  return pert_out

def mifgsm_attack(input,epsilon,data_grad):
  iter=10
  decay_factor=1.0
  pert_out = input
  alpha = epsilon/iter
  g=0.1
  for i in range(iter-1):
    g = decay_factor*g + data_grad/torch.norm(data_grad,p=1)
    pert_out = pert_out + alpha*torch.sign(g)
    pert_out = torch.clamp(pert_out, 0, 1)
    if torch.norm((pert_out-input),p=float('inf')) > epsilon:
      break
  return pert_out

# Training the model
# writer = SummaryWriter()

num_batches = len(train_loader)  
num_epochs = 2

num_steps = num_batches * num_epochs




attention_values_array = np.zeros((2*num_steps,20))
# attention_values_array_adv = np.zeros((num_steps,1))


optimizer = optim.SGD(model.parameters(), lr=0.01)
def train(epoch,epsilon,attack_state):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data.requires_grad = True
        optimizer.zero_grad()
        output, attention_values = model(data)
        attention_values_array[num_batches*(epoch-1)+batch_idx, :] = torch.sum(attention_values,[0,2,3]).detach().cpu().numpy()
        # # output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()        
        if attack_state:            
            model.zero_grad()
            # Collect datagrad
            data_grad = data.grad.data
            # epsilon = 0.6
            # Call FGSM Attack
            perturbed_data = fgsm_attack(data, epsilon, data_grad)
            output, attention_values = model(perturbed_data)   
            attention_values_array[num_steps+num_batches*(epoch-1)+batch_idx, :] = torch.sum(attention_values,[0,2,3]).detach().cpu().numpy()

            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()             
    return attention_values_array


#
# A simple test procedure to measure the STN performances on MNIST.
#
def test(epsilon,attack_state):
    #with torch.no_grad():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)            
        data.requires_grad = True
        output, attention_values = model(data)            
    # Calculate the loss
        loss = F.nll_loss(output, target)
    # Zero all existing gradients
        model.zero_grad()
    # Calculate gradients of model in backward pass
        loss.backward()
    # Collect datagrad
        data_grad = data.grad.data            
    # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad)          
        
        if attack_state:
            output, attention_values = model(perturbed_data)
        else:
            output, attention_values = model(data)
          # sum up batch loss
        test_loss += F.nll_loss(output, target, size_average=False).item()
        # get the index of the max log-probability
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)        
#    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
#          .format(test_loss, correct, len(test_loader.dataset),
#                  100. * correct / len(test_loader.dataset)))
    return correct
    
######################################################################
# Visualizing the STN results
# ---------------------------
epsilons = [0, .2, 0.4, 0.6, 0.8]

for epsilon in epsilons:
    for epoch in range(1, 2 + 1):
        train(epoch, epsilon, False)
        correct = test(epsilon, True)
    print('\nReg Train, Test set: Epsilon: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
      .format(epsilon, correct, len(test_loader.dataset),
              100. * correct / len(test_loader.dataset)))
    
    for epoch in range(1, 2 + 1):
        train(epoch, epsilon, True)
        correct = test(epsilon, True)
    print('\nAdv Train, Test set: Epsilon: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
      .format(epsilon, correct, len(test_loader.dataset),
              100. * correct / len(test_loader.dataset)))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    
    num_rows, num_cols = attention_values_array.shape

# تولید مختصات x، y و z بر اساس ابعاد ماتریس
    x = np.arange(num_cols)
    y = np.arange(num_rows)
    x, y = np.meshgrid(x, y)
    z = attention_values_array

    ax.plot_surface(x, y, z, cmap='viridis')

# تنظیم محورها
    ax.set_xlabel('محور X')
    ax.set_ylabel('محور Y')
    ax.set_zlabel('محور Z')

    # plt.scatter(attention_values_array)

    # num_rows, num_cols = attention_values_array.shape
    # fig, ax = plt.subplots(nrows=num_rows, ncols=num_cols)    
    # for i in range(num_cols):
    #     ax[i].plot(attention_values_array[i])
    #     ax[i].set_title("نمودار سطر {}".format(i))

    # plt.tight_layout()

    # plt.plot(attention_values_array)
    # plt.xlabel("Step")
    # plt.ylabel("Attention Value")
    # plt.title(f'epsilon{epsilon}')
    plt.show()



# .