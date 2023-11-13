import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#import torch
#from torch import nn
#import torch.nn.functional as F
#import torch.optim as optim
#from torch.utils.data import TensorDataset, DataLoader
#import torchvision
#import torchvision.transforms as transforms
#from torchvision.utils import make_grid
# additional torch packages
#import torch.nn.init as init
#from torch.nn import Linear, Conv2d, BatchNorm2d, MaxPool2d, Dropout2d
#from torch.nn.functional import relu, elu, relu6, sigmoid, tanh, softmax
from sklearn import metrics
#from torchvision.transforms import ToTensor
import rasterio
from rasterio.plot import show
import os
from glob import glob
from PIL import Image
import time


my_list = np.ones(5)*5
print(my_list)