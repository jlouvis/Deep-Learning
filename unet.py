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

# Set random seed for reproducibility
np.random.seed(42)

# Load images
path = 'data'

images_tensors = []
for subdirectory in os.listdir(path):
    subdirectory_path = os.path.join(path, subdirectory)
    single_image = Image.open(subdirectory_path)
    single_image = ToTensor()(single_image)
    images_tensors.append(single_image)


# Load labels
path = 'labels'

labels_tensors = []
for subdirectory in os.listdir(path):
    subdirectory_path = os.path.join(path, subdirectory)
    single_label = Image.open(subdirectory_path)
    single_label = ToTensor()(single_label)
    labels_tensors.append(single_label)

print("loaded images and labels")
first_10_images = images_tensors[:10]  # Get the first 10 images

# Create a figure with tight layout
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
plt.tight_layout()
# Iterate through the images and display them in the subplots with titles
for i, ax in enumerate(axes.flat):
    if i < len(first_10_images):
        image = first_10_images[i][0]  # Remove the extra dimension
        ax.imshow(image, cmap='gray')  # Use 'cmap' to specify the colormap (e.g., 'gray' for grayscale)
        ax.axis('off')  # Turn off the axis labels
        ax.set_title(f"Image {i+1}", fontsize=12)  # Add a title with "Image 1", "Image 2", etc.

# Show the plot
# save figure in Figure folder
#plt.savefig('figures/first_10_images.png')

label_images = labels_tensors[:10]  # Get the first 10 label images

# Create a figure with tight layout
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
plt.tight_layout()

# Iterate through the label images and display them in the subplots with titles
for i, ax in enumerate(axes.flat):
    if i < len(label_images):
        image = label_images[i][0]  # Remove the extra dimension
        ax.imshow(image, cmap='gray')  # Use 'cmap' to specify the colormap (e.g., 'gray' for grayscale)
        ax.axis('off')  # Turn off the axis labels
        ax.set_title(f"Label {i+1}", fontsize=12)  # Add a title with "Image 1", "Image 2", etc.

# save figure in Figure folder
#plt.savefig('figures/first_10_labels.png')

print("images and labels plotted")

# Data transformation

# # convert images in images_tensors to Float32
# images_tensors = [image.type(torch.FloatTensor) for image in images_tensors]
# # convert labels in labels_tensors to LongTensor
# labels_tensors = [label.type(torch.LongTensor) for label in labels_tensors]

# # remove channel dimension from labels - required for CrossEntropyLoss
# labels_tensors = [label[0] for label in labels_tensors]

# # Normalize images_tensors
# max_pixel = torch.max(torch.stack(images_tensors))
# mean_pixel = torch.mean(torch.stack(images_tensors[0:400]))
# std_pixel = torch.std(torch.stack(images_tensors[0:400]))
# images_norm = []
# for image in images_tensors:
#     image_norm = (image - mean_pixel) / std_pixel
#     images_norm.append(image_norm)

# Find the maximum pixel value in your dataset
max_pixel_value = torch.max(torch.stack(images_tensors))

images_norm = []
for image in images_tensors:
    # Normalize each pixel value to be between -1 and 1
    image_normalized = image / max_pixel_value
    images_norm.append(image_normalized)

# Reshape labels_tensors
labels_norm = []
for label in labels_tensors:
    # convert label to LongTensor - required for CrossEntropyLoss
    label_long = label.type(torch.LongTensor)
    labels_norm.append(label_long)


# remove channel dimension from labels - required for CrossEntropyLoss
labels_norm = [label[0] for label in labels_norm]

print("label shape: ",labels_norm[0].shape)
print("image shape: ",images_norm[0].shape)

# Build data sets

# Build Tensor dataset
dataset = TensorDataset(torch.stack(images_norm), torch.stack(labels_norm))

# Split in train (80%), validation (10%) and test (10%) sets
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

# batch size
batch_size = 5

# Build data loader
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True) # shuffle training set
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False) # no need to shuffle validation set
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False) # no need to shuffle test set

print("data sets built")

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

# Build U-Net model
# with batch normalization

class Unet(nn.Module):
    def __init__(self, n_class):
        super(Unet, self).__init__()

        # Encoder (downsampling)
        self.e11 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(64)
        self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2d(128)
        self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm2d(256)
        self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn41 = nn.BatchNorm2d(512)
        self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.bn51 = nn.BatchNorm2d(1024)
        self.e52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.bn52 = nn.BatchNorm2d(1024)

        # Decoder (upsampling)
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.bn11d = nn.BatchNorm2d(512)
        self.d12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm2d(512)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm2d(256)
        self.d22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn22d = nn.BatchNorm2d(256)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn31d = nn.BatchNorm2d(128)
        self.d32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm2d(128)

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d41 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn41d = nn.BatchNorm2d(64)
        self.d42 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn42d = nn.BatchNorm2d(64)

        # Output layer
        self.outconv = nn.Conv2d(64, n_class, kernel_size=1)

    def forward(self, x):
        # Encoder
        xe11 = F.relu(self.bn11(self.e11(x)))
        xe12 = F.relu(self.bn12(self.e12(xe11)))
        xp1 = self.pool1(xe12)

        xe21 = F.relu(self.bn21(self.e21(xp1)))
        xe22 = F.relu(self.bn22(self.e22(xe21)))
        xp2 = self.pool2(xe22)

        xe31 = F.relu(self.bn31(self.e31(xp2)))
        xe32 = F.relu(self.bn32(self.e32(xe31)))
        xp3 = self.pool3(xe32)

        xe41 = F.relu(self.bn41(self.e41(xp3)))
        xe42 = F.relu(self.bn42(self.e42(xe41)))
        xp4 = self.pool4(xe42)

        xe51 = F.relu(self.bn51(self.e51(xp4)))
        xe52 = F.relu(self.bn52(self.e52(xe51)))

        # Decoder
        xu1 = self.upconv1(xe52)
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd11 = F.relu(self.bn11d(self.d11(xu11)))
        xd12 = F.relu(self.bn12d(self.d12(xd11)))

        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xu2, xe32[:, :, :xu2.size(2), :xu2.size(3)]], dim=1)
        xd21 = nn.functional.relu(self.bn21d(self.d21(xu22)))
        xd22 = nn.functional.relu(self.bn22d(self.d22(xd21)))

        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xu3, xe22[:, :, :xu3.size(2), :xu3.size(3)]], dim=1)
        xd31 = F.relu(self.bn31d(self.d31(xu33)))
        xd32 = F.relu(self.bn32d(self.d32(xd31)))

        xu4 = self.upconv4(xd32)
        xu44 = torch.cat([xu4, xe12[:, :, :xu4.size(2), :xu4.size(3)]], dim=1)
        xd41 = F.relu(self.bn41d(self.d41(xu44)))
        xd42 = F.relu(self.bn42d(self.d42(xd41)))

        # Output layer
        out = self.outconv(xd42)

        return out

net = Unet(3) # 3 classes
if use_cuda:
    net.cuda()

device = torch.device("cuda" if use_cuda else "cpu")  # use cuda or cpu
net.to(device)
print(net)

# loss function: Cross entropy loss (for multi-class classification)
loss_unet = nn.CrossEntropyLoss()

# optimizer: ADAM
optimizer_unet = optim.Adam(net.parameters(), lr=1e-3)

# Train the model

# number of epochs to train the model
n_epochs = 10

# Steps
train_steps = len(train_set)//batch_size
val_steps = len(val_set)//batch_size
test_steps = len(test_set)//batch_size

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
        
        # Complete forward pass through model
        output = net(images)
        
        # Compute the loss
        train_loss = loss_unet(output, labels)

        # clean up gradients from previous run
        optimizer_unet.zero_grad()

        # Compute gradients using back propagation
        train_loss.backward()

        # Take a step with the optimizer to update the weights
        optimizer_unet.step()

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

            # Complete forward pass through model
            output = net(images)

            # Compute the loss
            val_loss = loss_unet(output, labels)

            # add the loss to the validation set's running loss 
            total_val_loss += val_loss

    # print training/validation statistics
    avg_train_loss = total_train_loss/train_steps
    avg_val_loss = total_val_loss/val_steps

    # update training history
    #train_losses.append(avg_train_loss.cpu().numpy())
    train_losses.append(avg_train_loss.cpu().detach().numpy())
    validation_losses.append(avg_val_loss.cpu().numpy())

    # print training/validation statistics
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch+1, avg_train_loss, avg_val_loss))
    
# display time elapsed for epoch
endtime = time.time()
print(f"Elapsed time: {(endtime - starttime)/60:.2f} min")
print("Finished Training")

