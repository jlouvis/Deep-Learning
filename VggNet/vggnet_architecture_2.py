import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid
# additional torch packages
import torch.nn.init as init
from torch.nn import Linear, Conv2d, BatchNorm2d, MaxPool2d, Dropout2d
from torch.nn.functional import relu, elu, relu6, sigmoid, tanh, softmax
from torchvision.transforms import ToTensor

class VGGnet(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        # input size: (1, 501, 501)
        #block 1
        self.e11 = Conv2d(1, 64, kernel_size=3, padding=1) # output: (64, 499, 499)
        self.e12 = Conv2d(64, 64, kernel_size=3, padding=1) # output: (64, 497, 497)
        self.pool1 = MaxPool2d(kernel_size=2, stride=2) # output: (64, 249, 249)
        #block2
        self.e21 = Conv2d(64, 128, kernel_size=3, padding=1) # output: (128, 247, 247)
        self.e22 = Conv2d(128, 128, kernel_size=3, padding=1) # output: (128, 245, 245)
        self.pool2 = MaxPool2d(kernel_size=2, stride=2) # output: (128, 123, 122)
        #block3
        self.e31 = Conv2d(128, 256, kernel_size=3, padding=1) # output: (256, 121, 120)
        self.e32 = Conv2d(256, 256, kernel_size=3, padding=1) # output: (256, 119, 118)
        self.e33 = Conv2d(256, 256, kernel_size=3, padding=1) # output: (256, 117, 116)
        self.pool3 = MaxPool2d(kernel_size=2, stride=2) # output: (256, 59, 58)
        #block4
        self.e41 = Conv2d(256, 512, kernel_size=3, padding=1) # output: (512, 57, 56)
        self.e42 = Conv2d(512, 512, kernel_size=3, padding=1) # output: (512, 55, 54)
        self.e43 = Conv2d(512, 512, kernel_size=3, padding=1) # output: (512, 53, 52)
        self.pool4 = MaxPool2d(kernel_size=2, stride=2) # output: (512, 27, 26)
        #block5
        self.e51 = Conv2d(512, 512, kernel_size=3, padding=1) # output: (512, 25, 24)
        self.e52 = Conv2d(512, 512, kernel_size=3, padding=1) # output: (512, 23, 22)
        self.e53 = Conv2d(512, 512, kernel_size=3, padding=1) # output: (512, 21, 20)
        self.pool5 = MaxPool2d(kernel_size=2, stride=2) # output: (512, 11, 10)
        # Transposed Convolutional Layers (Decoder - Upsampling)
        self.upconv5 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)   #output(512, 30, 30)
        self.d51 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.d52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d41 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.d42 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.d32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.d22 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.upconv = nn.ConvTranspose2d(64,64,kernel_size=2, stride=2)
        # Output layer
        self.outconv = nn.Conv2d(64, n_class, kernel_size=1)

    def forward(self,x):
        
        #block1
        x=F.relu(self.e11(x))
        x=F.relu(self.e12(x))
        x=self.pool1(x)
        #block2
        x=F.relu(self.e21(x))
        x=F.relu(self.e22(x))
        x=self.pool2(x)
        #block3
        x=F.relu(self.e31(x))
        x=F.relu(self.e32(x))
        x=F.relu(self.e33(x))
        x=self.pool3(x)
        #block4
        x=F.relu(self.e41(x))
        x=F.relu(self.e42(x))
        x=F.relu(self.e43(x))
        x=self.pool4(x)
        #block5
        x=F.relu(self.e51(x))
        x=F.relu(self.e52(x))
        x=F.relu(self.e53(x))
        x=self.pool5(x)
        # Transposed Convolutional Layers (Decoder - Upsampling)
        xu5 = self.upconv5(x)
        xu51 = torch.cat([xu5, x[:, :512, :xu5.size(2), :xu5.size(3)].repeat(1, 1, 2, 2)], dim=1)
        xd51 = F.relu(self.d51(xu51))
        xd52 = F.relu(self.d52(xd51))

        xu4 = self.upconv4(xd52)
        xu41 = torch.cat([xu4, xu51[:, :256, :xu4.size(2), :xu4.size(3)].repeat(1, 1, 2, 2)], dim=1)
        xd41 = F.relu(self.d41(xu41))
        xd42 = F.relu(self.d42(xd41))

        xu3 = self.upconv3(xd42)
        xu31 = torch.cat([xu3, xu41[:, :128, :xu3.size(2), :xu3.size(3)].repeat(1, 1, 2, 2)], dim=1)
        xd31 = F.relu(self.d31(xu31))
        xd32 = F.relu(self.d32(xd31))

        xu2 = self.upconv2(xd32)
        xu21 = torch.cat([xu2, xu31[:, :64, :xu2.size(2), :xu2.size(3)].reeat(1, 1, 2, 2)], dim=1)
        xd21 = F.relu(self.d21(xu21))
        xd22 = F.relu(self.d22(xd21))
        x=self.upconv(xd22)
        # Output layer
        out = self.outconv(x)
        return out


