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
# # import unet
# import unet_architecture
from unet_architecture_batchnorm_L2 import Unet_batch_dropout as Unet
from noise_function import add_noise_to_tensor as add_noise_to_tensor

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

net = Unet(n_class=3)
if use_cuda:
    net.cuda()

device = torch.device("cuda" if use_cuda else "cpu")  # use cuda or cpu
net.to(device)


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

# Define data augmentation transformations
hflip  = transforms.Compose([
    transforms.RandomHorizontalFlip(),   # Randomly flip the image horizontally
    
])

# Define data augmentation transformations
vflip  = transforms.Compose([
    transforms.RandomVerticalFlip(),   # Randomly flip the image horizontally
    
])

# Define data augmentation transformations
rot  = transforms.Compose([
    transforms.RandomRotation(degrees=90), # Randomly rotate the image by a certain degree
])


# Apply transformations to both images and labels simultaneously
hflip_data = [(hflip(image), hflip(label)) for image, label in zip(images_tensors, labels_tensors)]
vflip_data = [(vflip(image), vflip(label)) for image, label in zip(images_tensors, labels_tensors)]
rot_data = [(rot(image), rot(label)) for image, label in zip(images_tensors, labels_tensors)]


# Add augmented data to original data
images_tensors.extend([image for image, label in hflip_data])
labels_tensors.extend([label for image, label in hflip_data])
images_tensors.extend([image for image, label in vflip_data])
labels_tensors.extend([label for image, label in vflip_data])
images_tensors.extend([image for image, label in rot_data])
labels_tensors.extend([label for image, label in rot_data])

# Convert lists to tensors
images_tensors = torch.stack(images_tensors)
labels_tensors = torch.stack(labels_tensors)

# Crop images and labels to 128x128 and convert images to float
transform = transforms.CenterCrop((128, 128))
images_cropped = [transform(image) for image in images_tensors]
images_256 = [image.type(torch.FloatTensor) for image in images_cropped]
labels_cropped = [transform(label) for label in labels_tensors]

# convert labels to be 0, 1, 2
labels_cropped = [label * 2 for label in labels_cropped]
# convert labels to be long tensor
labels_cropped = [label.long() for label in labels_cropped]
# remove the third dimension from labels
labels_cropped = [label.squeeze(0) for label in labels_cropped]

max_pixel = torch.max(torch.stack(images_256[0:400]))
# Normalize the images
images_normalized = [image / max_pixel for image in images_256]

# Build data sets

# Build Tensor dataset
dataset = TensorDataset(torch.stack(images_normalized), torch.stack(labels_cropped))

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

# loss function: Cross entropy loss
loss_unet = nn.CrossEntropyLoss()

# optimizer: ADAM
optimizer_unet = optim.Adam(net.parameters(), lr=1e-5, weight_decay=1e-5)

# number of epochs to train the model
n_epochs = 50

# function to calculate accuracy
def accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    total = labels.size(0) * labels.size(1) * labels.size(2)
    correct = (predicted == labels).sum().item()
    return correct / total

# Initialize lists to store training and validation metrics
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

for epoch in range(n_epochs):
    # Training
    net.train()
    train_loss = 0.0
    train_acc = 0.0
    for images, labels in train_loader:
        images, labels = get_variable(images), get_variable(labels)

        noisy_images = add_noise_to_tensor(images)


        optimizer_unet.zero_grad()

        outputs = net(noisy_images)
        loss = loss_unet(outputs, labels)
        loss.backward()
        optimizer_unet.step()

        train_loss += loss.item() * images.size(0)
        train_acc += accuracy(outputs, labels) * images.size(0)

    train_loss = train_loss / len(train_loader.dataset)
    train_acc = train_acc / len(train_loader.dataset)

    # Store training metrics
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    # Validation
    net.eval()
    val_loss = 0.0
    val_acc = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = get_variable(images), get_variable(labels)
            outputs = net(images)
            loss = loss_unet(outputs, labels)

            val_loss += loss.item() * images.size(0)
            val_acc += accuracy(outputs, labels) * images.size(0)

    val_loss = val_loss / len(val_loader.dataset)
    val_acc = val_acc / len(val_loader.dataset)

    # Store validation metrics
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    print(f'Epoch [{epoch + 1}/{n_epochs}]')
    print(f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.4f}')
    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')

# Plot the train and validation loss
plt.figure(figsize=(10, 5))
plt.plot(range(1, n_epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, n_epochs+1), val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train and Validation Loss')
plt.legend()
# show val loss as text on plot for last epoch
plt.text(n_epochs, val_losses[-1], f'{val_losses[-1]:.4f}')
plt.savefig('figures/unet_standard_loss.png')

# Plot the train and validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(range(1, n_epochs+1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, n_epochs+1), val_accuracies, label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Train and Validation Accuracy')
plt.legend()
# show val accuracy as text on plot for last epoch
plt.text(n_epochs, val_accuracies[-1], f'{val_accuracies[-1]:.4f}')
plt.savefig('figures/unet_standard_accuracy.png')

# plot 1 image from validation set with original, predicted, ground truth labels and (predicted - ground truth) labels
net.eval()
images, labels = next(iter(val_loader))
images, labels = get_variable(images), get_variable(labels)
outputs = net(images)
_, predicted = torch.max(outputs, 1)
predicted = predicted.squeeze(1)
predicted = predicted.cpu().numpy()
labels = labels.squeeze(1)
labels = labels.cpu().numpy()
images = images.cpu().numpy()
images = np.transpose(images, (0, 2, 3, 1))

fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(20, 5))
axes[0].imshow(images[0], cmap='gray')
axes[0].set_title('Original Image')
axes[1].imshow(labels[0], cmap='gray')
axes[1].set_title('Ground Truth Label')
axes[2].imshow(predicted[0], cmap='gray')
axes[2].set_title('Predicted Label')
axes[3].imshow(np.abs(labels[0] - predicted[0]), cmap='gray')
axes[3].set_title('Difference')
# visualize the confusion matrix for the predicted and ground truth labels
confusion_matrix = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        confusion_matrix[i, j] = np.sum((predicted == i) & (labels == j))
axes[4].imshow(confusion_matrix)
# show actual numbers in confusion matrix
for i in range(3):
    for j in range(3):
        axes[4].text(j, i, int(confusion_matrix[i, j]), ha="center", va="center", color="red")
axes[4].set_title('Confusion Matrix')
axes[4].set_xlabel('Predicted Label')
axes[4].set_ylabel('Ground Truth Label')
plt.tight_layout()
plt.savefig('figures/unet_standard.png')