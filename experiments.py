################################################################################
# For now, I don't have too many flags in the script. We can comment/uncomment #
# pieces of code # as we need them. I tried to identify pieces as best as I    #
# could.                                                                       #
################################################################################

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
import os, operator

submission = False
validation = False

# Any results you write to the current directory are saved as output.
# Files are stored in pickle format.
# Load them like how you load any pickle. The data is a numpy array
train_images = pd.read_pickle('train_images.pkl')
train_labels = pd.read_csv('train_labels.csv')

test_images = pd.read_pickle('test_images.pkl')

# Hyper-parameters
num_epochs = 20
# 0123456789
num_classes = 10
# I found this to be easier on the memory
batch_size = 100
learning_rate = 0.001


# On kernel sizes... I am not too sure what kernel size to use. I went with 10
# to start with, but some blog posts suggest that smaller 3x3 sizes work better.
# Might be worth tweaking these things:
## https://towardsdatascience.com/deep-learning-3-more-on-cnns-handling-overfitting-2bd5d99abe5d

class TwoLayerNet(torch.nn.Module):
    def __init__(self):
        super(TwoLayerNet, self).__init__() # intialize recursively
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=10, stride=1, padding=2),
            nn.BatchNorm2d(32),
            # nn.Conv2d(1, 32, 16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=5, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 512 , kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # self.layer6 = nn.Sequential(
        #     nn.Conv2d(512, 512 , kernel_size=5, stride=1, padding=2),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU()
        #     # nn.MaxPool2d(kernel_size=2, stride=1)
        # )
        # Dropout is a regularization method.
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(512, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and
        return a tensor of output data. We can use
        Modules defined in the constructor as well as arbitrary
        operators on Variables.
        """
        # Piece together operations on tensors here
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        # out = selfelf.layer6(out)
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
# Normalize training data
for i in range(len(x)):
    x[i] = trans(x[i])
# Normalize testing data based on training mean and std
for i in range(len(x_test)):
    x_test[i] = trans(x_test[i])

x_train = x[0:30000]
y_train = y[0:30000]
x_val = x[30000:40000]
y_val = y[30000:40000]

# Create dataset and a loader to help with batching
train_dataset = torch.utils.data.TensorDataset(x_train,y_train)
val_dataset = torch.utils.data.TensorDataset(x_val,y_val)
test_dataset = torch.utils.data.TensorDataset(x_test)
# We will predict one testing example at a time
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

####################################################################
# This code is for preparring samples with replacement for bagging #
####################################################################
# samples = []

# make 100 samples with replacement
# for i in range(10):
#     train_sample = torch.utils.data.RandomSampler(train_dataset, replacement=True)
#     train_loader = torch.utils.data.DataLoader(train_dataset, sampler = train_sample, batch_size = batch_size)
#     samples.append(train_loader)

# # train 100 models on the 100 samples
# models = []

# NOTE: we may need to simplify the CNN to accelerate training. Or run on Trottier GPUs.

# for train_dataloader in samples:

# Separate data into training and validation set
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size)
validation_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size = 1)
model = TwoLayerNet()
# Loss function for classification tasks with C classes. Takes NxC tensor.
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)

total_step = len(train_dataloader)
loss_list = []
acc_list = []
epoch_number = 0
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

    torch.save(model, "epoch_cnn"+str(epoch_number)+".model")
    epoch_number = epoch_number + 1
    # # save trained model
    # models.append(model)

# # save trained models
# model_no = 0
# for model in models:
#     torch.save(model, "cnn_"+str(model_no)+".model")
#     model_no = model_no + 1

# models = []
#load trained models
# for i in range(10):
#     models.append(torch.load("cnn_"+str(i)+".model"))

def test_epoch_models(epochs, validation):
    for epoch_number in range(epochs):
        print("Testing on epoch " + str(epoch_number))
        model = torch.load("epoch_cnn"+str(epoch_number)+".model")
        validate_model([model], validation)

def validate_model(models, validation):
    errors = 0
    Id = 0
    for example in validation:
        votes = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}
        for model in models:
            # apply(lambda x : x.train(mode=False))
            model.train(mode=False)
            outputs = model(example[0])
            _, label_tensor = torch.max(outputs.data, 1)
            prediction = label_tensor[0].item()
            votes[prediction] = votes[prediction] + 1

        # print(votes)
        max_vote = max(votes.items(), key = operator.itemgetter(1))[0]
        # print(example)
        real_label = example[1]
        if(real_label.item() != max_vote):
            errors = errors + 1
        Id = Id + 1

    accuracy = 100.0 - errors/Id*100
    print("Validation accuracy: " + str(accuracy))

validate_model([model], validation_dataloader)
# Pass a list with a single model when using a single NN
def generate_submission(models, data):
    # produce submission
    with open("submission.csv", "w") as f:
        f.write("Id,Category\n")
        Id = 0
        for example in data:
            votes = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}
            for model in models:
                model.eval()
                outputs = model(example[0])
                _, label_tensor = torch.max(outputs.data, 1)
                prediction = label_tensor[0].item()
                votes[prediction] = votes[prediction] + 1

            # print(votes)
            max_vote = max(votes.items(), key = operator.itemgetter(1))[0]
            f.write(str(Id) + "," + str(max_vote) + "\n")
            Id = Id + 1
            if Id % 1000 == 0:
                print(Id)

# Write submission file
if submission:
    generate_submission(models, test_dataloader)
