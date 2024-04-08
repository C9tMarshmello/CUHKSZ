#### prepare data-set
#### you re expected to have either pickle or torchvision in your environment

import torch
from torch.utils.data import DataLoader, Dataset, Subset
import matplotlib.pyplot as plt

# python > 3.4
# if importlib.util.find_spec('torchvision') is None: # if you do not have torchvision, use TA's downloaded data
import pickle as pkl
with open('MNIST_dataset_local','rb') as read_file:
    data = pkl.load(read_file)
    read_file.close()
train_images, train_labels = data['train']
test_images, test_labels = data['test']
val_images, val_labels = train_images[5500:], train_labels[5500:]
train_images, train_labels = train_images[:5500], train_labels[:5500]
class myMnistDataSet(Dataset):
    def __init__(self,images,labels):
        self.X = images[:,None,...]
        self.y = labels
    def __len__(self):
        return len(self.X)
    def __getitem__(self, index):
        image = self.X[index]
        label = self.y[index]
        return torch.Tensor(image), torch.tensor(label)[0]
mnist_train = myMnistDataSet(train_images, train_labels.astype('int64'))
mnist_test = myMnistDataSet(test_images, test_labels.astype('int64'))
mnist_val = myMnistDataSet(val_images, val_labels.astype('int64'))

# else: # use torchvision
#     from torchvision import transforms, datasets
#
#     mnist_download_path = './MNIST_dataset'
#     datasets.MNIST(root=mnist_download_path, download=True)
#
#     mnist_train = datasets.MNIST(root = mnist_download_path, train = True, transform=transforms.ToTensor())
#     mnist_val   = Subset(mnist_train, list(range(5500,6000)))
#     mnist_train = Subset(mnist_train, list(range(5500)))
#
#     mnist_test  = datasets.MNIST(root = mnist_download_path, train = False, transform=transforms.ToTensor())


def get_loaders(batch_size):
    train_loader = DataLoader(mnist_train, batch_size = batch_size, shuffle=True)
    val_loader   = DataLoader(mnist_val, batch_size = batch_size, shuffle=False)
    test_loader  = DataLoader(mnist_test, batch_size = batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# get to know your loader
batch_size = 10
train_loader, val_loader, test_loader = get_loaders(batch_size)

for im, l in val_loader:
    break

print(im.shape, l.shape)
print(l)
import torch
import matplotlib.pyplot as plt
import numpy as np
# Function to get the average image for each label
def get_average_images(val_loader):
    averages = torch.zeros((10, 28, 28))
    counts = torch.zeros(10)

    with torch.no_grad():
        for im, l in val_loader:
            for i in range(len(l)):
                label = l[i].item()
                averages[label] += im[i][0]
                counts[label] += 1

    averages /= counts.view(-1, 1, 1)
    return averages

# Display the average images
def display_average_images(averages):
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(averages[i], cmap='gray')
        plt.title(f'Label {i}')
        plt.axis('off')
    plt.show()

# Call the functions
averages = get_average_images(val_loader)
display_average_images(averages)
