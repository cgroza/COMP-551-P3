import pickle as pkl
import torch
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

# Any results you write to the current directory are saved as output.
#Files are stored in pickle format.
#Load them like how you load any pickle. The data is a numpy array
import pandas as pd
train_images = pd.read_pickle('train_images.pkl')
train_labels = pd.read_csv('train_labels.csv')

import matplotlib.pyplot as plt

#Let's show image with id 16
img_idx = 16

plt.title('Label: {}'.format(train_labels.iloc[img_idx]['Category']))
plt.imshow(train_images[img_idx])


class TwoLayerNet(torch.nn.Module):
    def __init__(self):
        super(TwoLayerNet, self).__init__() # intialize recursively
        # I think out layers should be a series of convolutions
        # First convolution taking an input 64x64 and outputting into 10 nodes.
        self.conv = nn.Conv2d(40000, 10, (64, 64))
        # Long likely hood values for each class
        self.logsoftmax = torch.nn.LogSoftmax()

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and
        return a tensor of output data. We can use
        Modules defined in the constructor as well as arbitrary
        operators on Variables.
        """
        # Piece together operations on tensors here
        conv = self.conv(x)
        y_pred = self.logsoftmax(conv)
        return y_pred

# Fill x with data from training images
x = torch.from_numpy(train_images)
# Fill y with the labels
y = torch.from_numpy(train_labels['Category'].values)

model = TwoLayerNet()

# Loss function for classification tasks with C classes. Takes NxC tensor of
# log likely-hoods as input, and (N, 1) tensor as training labels.
criterion = torch.nn.NLLLoss()

# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
losses = []

for epoch in range(50):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x)

    # Compute and print loss
    loss = criterion(y_pred, y)
    losses.append(loss.data.item())
    print("Epoch : " + str(epoch))
    print("Loss : " + str(loss.data.item()))
    # Reset gradients to zero, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
