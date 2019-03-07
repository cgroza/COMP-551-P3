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
import matplotlib.pyplot as plt
import os

submission = False

# Any results you write to the current directory are saved as output.
#Files are stored in pickle format.
#Load them like how you load any pickle. The data is a numpy array
train_images = pd.read_pickle('train_images.pkl')
train_labels = pd.read_csv('train_labels.csv')

test_images = pd.read_pickle('test_images.pkl')


# Let's show image with id 16
# img_idx = 16
# plt.title('Label: {}'.format(train_labels.iloc[img_idx]['Category']))
# plt.imshow(train_images[img_idx])

# Hyperparameters
num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.001


class TwoLayerNet(torch.nn.Module):
    def __init__(self):
        super(TwoLayerNet, self).__init__() # intialize recursively
        # # I think out layers should be a series of convolutions
        # # First convolution taking an input 64x64 and outputting into 10 nodes.
        # self.conv = nn.Conv2d(40000, 10, (64, 64))
        # # Long likely hood values for each class
        # self.logsoftmax = torch.nn.LogSoftmax()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=10, stride=1, padding=2),
            # nn.Conv2d(1, 32, 16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=5, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # Dropout is a regularization method.
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(12544, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and
        return a tensor of output data. We can use
        Modules defined in the constructor as well as arbitrary
        operators on Variables.
        """
        # Piece together operations on tensors here
        # conv = self.conv(x)
        # y_pred = self.logsoftmax(conv)
        # return y_pred
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

# Fill x with data from training images
x = torch.from_numpy(train_images).reshape((40000, 1, 64, 64))
# Fill x_test with data from training images
x_test = torch.from_numpy(test_images).reshape((test_images.shape[0], 1, 64, 64))

# Fill y with the labels
y = torch.from_numpy(train_labels['Category'].values)

# Normalize Grayscale values. NNs work best when data ranges from [-1, 1]
trans = transforms.Compose([transforms.Normalize((torch.mean(x),), (torch.std(x),))])

for i in range(len(x)):
    x[i] = trans(x[i])

for i in range(len(x_test)):
    x_test[i] = trans(x_test[i])

# Create dataset and a loader to help with batching

train_dataset = torch.utils.data.TensorDataset(x,y)
test_dataset = torch.utils.data.TensorDataset(x_test)

test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

samples = []
# make 100 samples with replacement
for i in range(100):
    train_sample = torch.utils.data.RandomSampler(train_dataset, replacement=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, sampler = train_sample, batch_size = batch_size)
    samples.append(train_loader)

# train 100 models on the 100 samples
models = []

# NOTE: we may need to simplify the CNN to accelerate training. Or run on Trottier GPUs.
for train_dataloader in samples:
    model = TwoLayerNet()

    # Loss function for classification tasks with C classes. Takes NxC tensor.

    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    total_step = len(train_dataloader)
    loss_list = []
    acc_list = []
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_dataloader):
            # i is a batch number here. images contains 100 training examples.
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_list.append(loss.item())

            # Back-propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                    .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                            (correct / total) * 100))
    # save trained model
    models.append(model)

# TODO: change this function to take vote by majority from 100 models.
def generate_submission(model, data):
    # produce submission
    with open("submission.csv", "w") as f:
        f.write("Id,Category\n")
        Id = 0
        for example in data:
            # print(example[0])
            outputs = model(example[0])
            _, prediction = torch.max(outputs.data, 1)
            # print(prediction[0].item())
            # print(type(prediction))
            f.write(str(Id) + "," + str(prediction[0].item()) + "\n")
            Id = Id + 1


if submission:
    generate_submission(model, test_dataloader)
