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

class Unet(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        # Encoder (downsampling)
        # input size: (1, 501, 501)
        self.e11 = Conv2d(1, 64, kernel_size=3, padding=1) # output: (64, 499, 499)
        self.e12 = Conv2d(64, 64, kernel_size=3, padding=1) # output: (64, 497, 497)
        self.pool1 = MaxPool2d(kernel_size=2, stride=2) # output: (64, 249, 249)

        # input size: (64, 249, 249)
        self.e21 = Conv2d(64, 128, kernel_size=3, padding=1) # output: (128, 247, 247)
        self.e22 = Conv2d(128, 128, kernel_size=3, padding=1) # output: (128, 245, 245)
        self.pool2 = MaxPool2d(kernel_size=2, stride=2) # output: (128, 122, 122)

        # input size: (128, 122, 122)
        self.e31 = Conv2d(128, 256, kernel_size=3, padding=1) # output: (256, 120, 120)
        self.e32 = Conv2d(256, 256, kernel_size=3, padding=1) # output: (256, 118, 118)
        self.pool3 = MaxPool2d(kernel_size=2, stride=2) # output: (256, 59, 59)

        # input size: (256, 59, 59)
        self.e41 = Conv2d(256, 512, kernel_size=3, padding=1) # output: (512, 57, 57)
        self.e42 = Conv2d(512, 512, kernel_size=3, padding=1) # output: (512, 55, 55)
        self.pool4 = MaxPool2d(kernel_size=2, stride=2) # output: (512, 27, 27)

        # input size: (512, 27, 27)
        self.e51 = Conv2d(512, 1024, kernel_size=3, padding=1) # output: (1024, 25, 25)
        self.e52 = Conv2d(1024, 1024, kernel_size=3, padding=1) # output: (1024, 23, 23)

        # Decoder (upsampling)

        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.d11 = Conv2d(1024, 512, kernel_size=3, padding=1)
        self.d12 = Conv2d(512, 512, kernel_size=3, padding=1)
        

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d21 = Conv2d(512, 256, kernel_size=3, padding=1)
        self.d22 = Conv2d(256, 256, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d31 = Conv2d(256, 128, kernel_size=3, padding=1)
        self.d32 = Conv2d(128, 128, kernel_size=3, padding=1)

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d41 = Conv2d(128, 64, kernel_size=3, padding=1)
        self.d42 = Conv2d(64, 64, kernel_size=3, padding=1)

        # Output layer
        self.outconv = Conv2d(64, n_class, kernel_size=1)


    def forward(self, x):
        # Encoder
        xe11 = relu(self.e11(x))
        xe12 = relu(self.e12(xe11))
        xp1 = self.pool1(xe12)

        xe21 = relu(self.e21(xp1))
        xe22 = relu(self.e22(xe21))
        xp2 = self.pool2(xe22)

        xe31 = relu(self.e31(xp2))
        xe32 = relu(self.e32(xe31))
        xp3 = self.pool3(xe32)

        xe41 = relu(self.e41(xp3))
        xe42 = relu(self.e42(xe41))
        xp4 = self.pool4(xe42)

        xe51 = relu(self.e51(xp4))
        xe52 = relu(self.e52(xe51))

        # Decoder
        xu1 = self.upconv1(xe52)
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd11 = relu(self.d11(xu11))
        xd12 = relu(self.d12(xd11))

        xu2 = self.upconv2(xd12)
        #xu22 = torch.cat([xu2, xe32], dim=1)
        xu22 = torch.cat([xu2, xe32[:, :, :xu2.size(2), :xu2.size(3)]], dim=1)
        xd21 = relu(self.d21(xu22))
        xd22 = relu(self.d22(xd21))

        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xu3, xe22[:, :, :xu3.size(2), :xu3.size(3)]], dim=1)
        xd31 = relu(self.d31(xu33))
        xd32 = relu(self.d32(xd31))

        xu4 = self.upconv4(xd32)
        xu44 = torch.cat([xu4, xe12[:, :, :xu4.size(2), :xu4.size(3)]], dim=1)
        xd41 = relu(self.d41(xu44))
        xd42 = relu(self.d42(xd41))

        # Output layer
        out = self.outconv(xd42)

        return out

