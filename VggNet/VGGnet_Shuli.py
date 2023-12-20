#matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
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
#from sklearn import metrics
from torchvision.transforms import ToTensor
import os
from glob import glob
from PIL import Image
import time

# Set random seed for reproducibility
np.random.seed(42)

path = 'data'

images_tensors = []
for subdirectory in os.listdir(path):
    subdirectory_path = os.path.join(path, subdirectory)
    single_image = Image.open(subdirectory_path)
    single_image = ToTensor()(single_image)
    images_tensors.append(single_image)

print(len(images_tensors))
print(type(images_tensors))
print(type(images_tensors[0]))
print(images_tensors[0].shape)

path = 'labels'

labels_tensors = []
for subdirectory in os.listdir(path):
    subdirectory_path = os.path.join(path, subdirectory)
    single_label = Image.open(subdirectory_path)
    single_label = ToTensor()(single_label)
    labels_tensors.append(single_label)

print(len(labels_tensors))
print(type(labels_tensors))
print(type(labels_tensors[0]))
print(labels_tensors[0].shape)

len(labels_tensors[0].unique())

transform = transforms.CenterCrop((256, 256))
images_cropped = [transform(image) for image in images_tensors]

transform = transforms.CenterCrop((256, 256))
images_cropped = [transform(image) for image in images_tensors]
# convert images_256 to float tensor
images_256 = [image.type(torch.FloatTensor) for image in images_cropped]
labels_256 = [transform(label) for label in labels_tensors]

labels_256[0].shape

# Convert labels to one-hot encoding
labels_one_hot = [F.one_hot(label.squeeze().long(), num_classes=3).permute(2, 0, 1).float() for label in labels_256]

# Stack the one-hot encoded labels together
labels_stacked = torch.stack(labels_one_hot)

print(labels_stacked.shape)

labels_stacked[2].unique()

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

print("Number of train images and labels: ", len(train_set))
print("Number of validation images and labels: ", len(val_set))
print("Number of test images and labels: ", len(test_set))

# batch size
batch_size = 8

# Build data loader
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True) # shuffle training set
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False) # no need to shuffle validation set
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False) # no need to shuffle test set

# print shape of first batch of images in train_loader
for images, labels in train_loader:
    print("Shape of images in a batch in train_loader: ", images.shape)
    print("Shape of labels in a batch in train_loader: ", labels.shape)
    break

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

net = VGGnet(3) # 3 classes
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)

net.apply(weights_init)


if use_cuda:
    net.cuda()

device = torch.device("cuda" if use_cuda else "cpu")  # use cuda or cpu


net.to(device)
print(net)

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, prediction, target):
        smooth = 1e-4

        intersection = torch.sum(prediction * target)
        union = torch.sum(prediction) + torch.sum(target)

        dice_coefficient = (2.0 * intersection + smooth) / (union + smooth)

        return 1.0 - dice_coefficient
    
# loss function: Cross entropy loss
loss_VGGnet =  nn.CrossEntropyLoss()

# optimizer: ADAM
optimizer_VGGnet = optim.Adam(net.parameters(), lr=1e-3)

# number of epochs to train the model
n_epochs = 25

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

            # Complete forward pass through model
            output = net(images)

            # Compute the loss
            val_loss = loss_VGGnet(output, labels)

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

plt.plot(np.linspace(1, n_epochs, 25), train_losses, 'b', label='Training loss')
plt.plot(np.linspace(1, n_epochs, 25), validation_losses, 'r', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('figures/loss.png')
