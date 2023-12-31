# import libraries
import numpy as np
import matplotlib.pyplot as plt
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
import os
from glob import glob
from PIL import Image
import time

path = 'data'

images_tensors = []
for subdirectory in os.listdir(path):
    subdirectory_path = os.path.join(path, subdirectory)
    single_image = Image.open(subdirectory_path)
    single_image = ToTensor()(single_image)
    images_tensors.append(single_image)

path = 'labels'

labels_tensors = []
for subdirectory in os.listdir(path):
    subdirectory_path = os.path.join(path, subdirectory)
    single_label = Image.open(subdirectory_path)
    single_label = ToTensor()(single_label)
    labels_tensors.append(single_label)

use_cuda = torch.cuda.is_available()
print("Running GPU.") if use_cuda else print("No GPU available.")


def get_variable(x):
    """ Converts tensors to cuda, if available. """
    if use_cuda:
        return x.cuda()
    return x


def get_numpy(x):
    """ Get numpy array for both cuda and not. """
    if use_cuda:
        return x.cpu().data.numpy()
    return x.data.numpy()

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
        xu51 = torch.cat([xu5, x[:, :xu5.size(1), :xu5.size(2), :xu5.size(3)].repeat(1, 1, 2, 2)], dim=1)
        xd51 = F.relu(self.d51(xu51))
        xd52 = F.relu(self.d52(xd51))

        xu4 = self.upconv4(xd52)
        xu41 = torch.cat([xu4, xu51[:, :xu4.size(1), :xu4.size(2), :xu4.size(3)].repeat(1, 1, 2, 2)], dim=1)
        xd41 = F.relu(self.d41(xu41))
        xd42 = F.relu(self.d42(xd41))

        xu3 = self.upconv3(xd42)
        xu31 = torch.cat([xu3, xu41[:, :xu3.size(1), :xu3.size(2), :xu3.size(3)].repeat(1, 1, 2, 2)], dim=1)
        xd31 = F.relu(self.d31(xu31))
        xd32 = F.relu(self.d32(xd31))

        xu2 = self.upconv2(xd32)
        xu21 = torch.cat([xu2, xu31[:, :xu2.size(1), :xu2.size(2), :xu2.size(3)].repeat(1, 1, 2, 2)], dim=1)
        xd21 = F.relu(self.d21(xu21))
        xd22 = F.relu(self.d22(xd21))

        # Output layer
        out = self.outconv(xd22)

        out = F.interpolate(out, size=(501, 501), mode='bilinear', align_corners=False)

        return out


net = VGGnet(3) # 3 classes
if use_cuda:
    net.cuda()

device = torch.device('cpu')  # use cuda or cpu
net.to(device)
print(net)

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, prediction, target):
        smooth = 1e-5

        intersection = torch.sum(prediction * target)
        union = torch.sum(prediction) + torch.sum(target)

        dice_coefficient = (2.0 * intersection + smooth) / (union + smooth)

        return 1.0 - dice_coefficient

# loss function: Cross entropy loss
loss_VGGnet = DiceLoss()

# optimizer: ADAM
optimizer_VGGnet = optim.Adam(net.parameters(), lr=1e-3)

train_images = images_tensors[:350]
train_labels = labels_tensors[:350]
val_images = images_tensors[350:400]
val_labels = labels_tensors[350:400]
test_images = images_tensors[400:]
test_labels = labels_tensors[400:]

# batch size
batch_size = 10

# Create Tensor datasets
train_data = TensorDataset(torch.stack(train_images), torch.stack(train_labels))
val_data = TensorDataset(torch.stack(val_images), torch.stack(val_labels))
test_data = TensorDataset(torch.stack(test_images), torch.stack(test_labels))

# Create DataLoaders
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Get one batch of training data
images, labels = next(iter(train_loader))

# Print the image tensor shape
print("Image tensor shape:", images.shape)

print(labels.shape)

import time

# number of epochs to train the model
n_epochs = 50

# Steps
train_steps = len(train_loader)//batch_size
val_steps = len(val_loader)//batch_size
test_steps = len(test_loader)//batch_size

# validation loss
valid_loss_min = np.Inf

# lists to store training and validation losses
train_losses = []
validation_losses = []

# lists to store training and validation accuracies
train_accuracies = []
validation_accuracies = []

net.train()

# start time (for printing elapsed time per epoch)
starttime = time.time()
for epoch in range(n_epochs):
    
    total_train_loss = 0
    total_val_loss = 0

    # loop over training data
    for images, labels in train_loader:
        # send the input to device
        images, labels = images.to(device), labels.to(device)
        images = images.to(torch.float32)
        labels = labels.to(torch.long)
        # Complete forward pass through model
        output = net(images)

        output_tensor = output.permute(0,2,3,1)
        output_flattened = output_tensor.reshape(-1, 3)
        labels_flattened = labels.view(-1)

        # Compute the loss
        train_loss = loss_VGGnet(output, labels)

        # clean up gradients from previous run
        optimizer_VGGnet.zero_grad()

        # Compute gradients using back propagation
        train_loss.backward()

        # Take a step with the optimizer to update the weights
        optimizer_VGGnet.step()

        # add the loss to the training set's running loss
        total_train_loss += train_loss

    # Turn off gradients for validation, saves memory and computations
    with torch.no_grad():
        # set the model to evaluation mode
        net.eval()

        # loop over validation data
        for images, labels in val_loader:
            # send the input to device
            images, labels = images.to(device), labels.to(device)
            images = images.to(torch.float32)
            labels = labels.to(torch.long)

            # Complete forward pass through model
            output = net(images)
            output_tensor = output.permute(0,2,3,1)
            output_flattened = output_tensor.reshape(-1, 3)
            labels_flattened = labels.view(-1)
            # Compute the loss
            val_loss = loss_VGGnet(output, labels)

            # add the loss to the validation set's running loss 
            total_val_loss += val_loss

    # print training/validation statistics
    avg_train_loss = total_train_loss/train_steps
    avg_val_loss = total_val_loss/val_steps

    # update training history
    train_losses.append(avg_train_loss.detach().cpu().numpy())
    validation_losses.append(avg_val_loss.detach().cpu().numpy())

    # print training/validation statistics
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch+1, avg_train_loss, avg_val_loss))
    
# display time elapsed for epoch
endtime = time.time()
print(f"Elapsed time: {(endtime - starttime)/60:.2f} min")
print("Finished Training")


