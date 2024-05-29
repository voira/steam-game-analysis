import matplotlib.pyplot as plt # for plotting
import numpy as np # for transformation
import pandas as pd
from PIL import Image
import os
import shutil
from os.path import exists

import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch # PyTorch package
import torchvision # load datasets
import torchvision.transforms as transforms # transform data
import torch.nn as nn # basic building block for neural neteorks
import torch.nn.functional as F # import convolution functions like Relu
import torch.optim as optim # optimzer

# Load the JSON data into a python dictionary
train_data = pd.read_json("train_data.json")

# Clean out the games that have no reviews
train_df = train_data.dropna(subset=["sentiment"])

# explode the dataset
train_df_expanded=train_df.explode("screenshots", ignore_index=True)

# Load the JSON data into a python dictionary
valid_data = pd.read_json("valid_data.json")

# Clean out the games that have no reviews
valid_df = valid_data.dropna(subset=["sentiment"])

# explode the dataset
valid_df_expanded=valid_df.explode("screenshots", ignore_index=True)

# Initialize the transformation funciton
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     
     ])

# training dataset
train_dataset = datasets.ImageFolder(
    root='train',
    transform=transform
)

# validation dataset
valid_dataset = datasets.ImageFolder(
    root='valid',
    transform=transform
)

# training data loaders
train_loader = DataLoader(
    train_dataset, batch_size=64, shuffle=True,
    num_workers=4, pin_memory=True
)

# validation data loaders
valid_loader = DataLoader(
    valid_dataset, batch_size=64, shuffle=False,
    num_workers=4, pin_memory=True
)