"""
EECS 445 - Introduction to Machine Learning
Winter 2023 - Project 2
Target CNN
    Constructs a pytorch model for a convolutional neural network
    Usage: from model.target import target
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from utils import config


class Target(nn.Module):
    def __init__(self):
        """Define the architecture, i.e. what layers our network contains. 
        At the end of __init__() we call init_weights() to initialize all model parameters (weights and biases)
        in all layers to desired distributions."""
        super().__init__()

        ## TODO: define each layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.fc_1 = nn.Linear(in_features=3136, out_features=64)
        self.fc_2 = nn.Linear(in_features=64, out_features=10)

        ##

        self.init_weights()

    def init_weights(self):
        """Initialize all model parameters (weights and biases) in all layers to desired distributions"""        
        torch.manual_seed(42)

        for conv in [self.conv1, self.conv2, self.conv3]:
            C_in = conv.weight.size(1) # gets # of input channels
            nn.init.normal_(conv.weight, 0.0, 1 / sqrt(3 * 3 * C_in)) # normal distributes it
            nn.init.constant_(conv.bias, 0.0) # gives bias of 0.0

        # fc_1
        nn.init.normal_(self.fc_1.weight, 0.0, sqrt(1 / self.fc_1.weight.size(1)))
        nn.init.constant_(self.fc_1.bias, 0.0)

        # fc_2
        nn.init.normal_(self.fc_2.weight, 0.0, sqrt(1 / self.fc_2.weight.size(1)))
        nn.init.constant_(self.fc_2.bias, 0.0)

        ##

    def forward(self, x):
        """This function defines the forward propagation for a batch of input examples, by
            successively passing output of the previous layer as the input into the next layer (after applying
            activation functions), and returning the final output as a torch.Tensor object

            You may optionally use the x.shape variables below to resize/view the size of
            the input matrix at different points of the forward pass"""

        # print(x.shape)
        ReLU = nn.ReLU()
        flatten = nn.Flatten()
        dropout = nn.Dropout()
        # print(x.shape)

        layer = self.pool(ReLU(self.conv1(x)))
        layer = self.pool(ReLU(self.conv2(layer)))
        layer = ReLU(self.conv3(layer))
        layer = dropout(self.fc_1(flatten(layer)))
        layer = self.fc_2(layer)
        return layer


        # layer_1_done = ReLU(self.conv1(x))
        # layer_2_done = self.pool(layer_1_done)
        # layer_3_done = ReLU(self.conv2(layer_2_done))
        # layer_4_done = self.pool(layer_3_done)
        # layer_5_done = ReLU(self.conv3(layer_4_done))
        # layer_6_done = self.fc_1(flatten(layer_5_done))

        # ##

        # return layer_6_done
