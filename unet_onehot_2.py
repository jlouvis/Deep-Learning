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
import torch

# Set random seed for reproducibility
np.random.seed(42)

image_path = 'data'
label_path = 'labels'

images_tensors = []
labels_tensors = []

# Getting a sorted list of filenames from both image and label directories
image_files = sorted(os.listdir(image_path))
label_files = sorted(os.listdir(label_path))

# Zipping the sorted filenames so they correspond to each other
for img_filename, lbl_filename in zip(image_files, label_files):
    if img_filename[9:12] == lbl_filename[7:10]:  # Matching filenames by position
        img_filepath = os.path.join(image_path, img_filename)
        lbl_filepath = os.path.join(label_path, lbl_filename)

        # Reading and converting images to tensors
        single_image = Image.open(img_filepath)
        single_image = ToTensor()(single_image)
        images_tensors.append(single_image)

        # Reading and converting labels to tensors
        single_label = Image.open(lbl_filepath)
        single_label = ToTensor()(single_label)
        labels_tensors.append(single_label)

print("loaded images and labels")

# Crop images and labels to 256x256 and convert images to float
transform = transforms.CenterCrop((256, 256))
images_cropped = [transform(image) for image in images_tensors]
images_256 = [image.type(torch.FloatTensor) for image in images_cropped]
labels_cropped = [transform(label) for label in labels_tensors]

# Convert labels to one-hot encoding
labels_one_hot = [F.one_hot(label.squeeze().long(), num_classes=3).permute(2, 0, 1).float() for label in labels_cropped]

# Stack the one-hot encoded labels together
labels_stacked = torch.stack(labels_one_hot)

max_pixel = torch.max(torch.stack(images_256[0:400]))
# Normalize the images
images_normalized = [image / max_pixel for image in images_256]

# Build data sets

# Build Tensor dataset
dataset = TensorDataset(torch.stack(images_normalized), labels_stacked)

# Split in train (80%), validation (10%) and test (10%) sets
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

# batch size
batch_size = 4

# Build data loader
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True) # shuffle training set
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False) # no need to shuffle validation set
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False) # no need to shuffle test set

# GPU
use_cuda = torch.cuda.is_available()
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

net = Unet(3) # 3 classes
if use_cuda:
    net.cuda()

device = torch.device("cuda" if use_cuda else "cpu")  # use cuda or cpu
net.to(device)

# loss function: Cross entropy loss
loss_unet = nn.CrossEntropyLoss()

# optimizer: ADAM
optimizer_unet = optim.Adam(net.parameters(), lr=1e-5)
#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_unet, 'min', patience=2)

# number of epochs to train the model
n_epochs = 50

# initialize lists to store losses and accuracies
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(n_epochs):
    # Training
    net.train()
    train_loss = 0
    correct_train = 0
    total_train = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer_unet.zero_grad()
        outputs = net(images)
        loss = loss_unet(outputs, labels.argmax(dim=1))
        loss.backward()
        optimizer_unet.step()

        # compute training loss
        train_loss += loss.item()

        # compute training accuracy
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        # correct train - ensure correct dimensions
        correct_train += (predicted == labels.argmax(dim=1)).sum().item()

    train_loss /= len(train_loader)
    train_accuracy = correct_train / total_train
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    # Validation
    net.eval()
    val_loss = 0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = net(images)
            loss = loss_unet(outputs, labels.argmax(dim=1))

            val_loss += loss.item()

            # compute validation accuracy
            _, predicted = torch.max(outputs, 1)
            total_val += labels.size(0)
            # correct val - ensure correct dimensions
            correct_val += (predicted == labels.argmax(dim=1)).sum().item()

    val_loss /= len(val_loader)
    val_accuracy = correct_val / total_val
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    # Print epoch results
    print(f"Epoch {epoch+1}/{n_epochs}:")
    print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracies[-1]:.2f}%")
    print(f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracies[-1]:.2f}%")
    print()

# Plot the train and validation loss
plt.figure(figsize=(10, 5))
plt.plot(range(1, n_epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, n_epochs+1), val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train and Validation Loss')
plt.legend()
plt.savefig('figures/loss.png')

# Plot the train and validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(range(1, n_epochs+1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, n_epochs+1), val_accuracies, label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Train and Validation Accuracy')
plt.legend()
plt.savefig('figures/accuracy.png')
