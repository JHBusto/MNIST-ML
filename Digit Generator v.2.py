import torch
import random
import torchvision
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

#Defining device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#Defining model parameters, these must stay constant or things will break due to model loading requiring these values
image_size = 784
hidden_size = 256
latent_size = 64

# Can be changed
batch_size = 1

# Structure of the decider
Decider = nn.Sequential(
    nn.Linear(image_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, latent_size),
    nn.Tanh())
Decider.to(device)

# Structure of the generator
G = nn.Sequential(
    nn.Linear(latent_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, image_size),
    nn.Tanh())
G.to(device);

# String is the path from which the dictionary containing the paranmeters to be loaded is
G.load_state_dict(torch.load("DigitGen300Epochs.pt"))

# Eval modifies the initialization after loading the parameters somehow
# We write it as the alternative is training, which is certainly not what we are after here
G.eval()


# This is stuff for saving images
from torchvision.utils import save_image
from IPython.display import Image
import os

# First we check to see if our sample directory exists and if not we make one
# Next we take our input vectors and run them trough the model, then reshape the 748 output into 1,28,28 tensors
# Finally, the tensors are formatted into png files and placed in rows of 10 before being saved to sample directory
def save_fake_images(index):
    sample_dir = 'Sample digits'
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    fake_images_raw = G(input_vectors)
    fake_images = fake_images_raw.reshape(fake_images_raw.size(0), 1, 28, 28)
    fake_fname = 'fake_images-{0:0=4d}.png'.format(index)
    print('Saving', fake_fname)
    save_image(fake_images, os.path.join(sample_dir, fake_fname), nrow=10)


# Input vectors for Generator created by Decider

# First, create random inputs for Decider
D_input_vectors = torch.randn(batch_size, image_size).to(device)

# Create the inputs for G by running the Decider
G_input_vectors = Decider(D_input_vectors)

sample = G(G_input_vectors)

Tensors = []
rating = 1000
Tensor = (sample, rating)
Tensors.append(Tensor)
print(sample)
print(Tensors)
#save_fake_images(0)

