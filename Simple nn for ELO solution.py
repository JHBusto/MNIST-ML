# The Rater trains by assessing itself according to the ratings of the images which the human has viewed where loss is
# defined by how close the guessed rating is to the true (human) rating of a given image
# True ratings are determined by images being compared by the human and receiving or losing rating according to the
# ELO rating system
# The human-rated images are placed into a list which serves as the training set for the Rater

# The loss function for the rater will have to be something like a sigmoidal or logarithmic function with limit of y=1
# based off how far away it guesses from the true rating.

# Input is image_size, since it will be fed images. Output size is 1 as we want a single number as our rating
# There might be some kind of upper limit to the values the nn can generate and this could create issues for the
# Elo rating process as it may artificially limit the sensitivity of the Rater. (ex. if a really good image is generated
# it may only give it a mediocre rating as it is unable to produce a value high enough to reflect the goodness of the
# image)

import torch
import random
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
from tkinter import *
from PIL import ImageTk
from torchvision.transforms import ToPILImage
from scipy import stats
import numpy as np
import threading


### Parameters and initializations ###

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Defining model parameters, image_size, hidden_size and latent_size must be 784, 256 and 64 respectively,
# because these are the parameters for the generator model we intend to import.
# If the model parameters are off, the imported weights will be placed wrong and the generator will not work.
image_size = 784
hidden_size = 256
latent_size = 64

# This is for the rater
rating_output_number = 1

# Can be changed
batch_size = 5
initialrating = float(100)

#Number of epochs
total_epochs = 100

# Structure of the Decider
Decidernn = nn.Sequential(
    nn.Linear(image_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, latent_size),
    nn.Tanh())
Decidernn.to(device)

MnistModel_optimizer_Decidernn = torch.optim.Adam(Decidernn.parameters(), lr=0.0002)

# This is the structure of the rater
Raternn = nn.Sequential(
    nn.Linear(image_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, rating_output_number))
Raternn.to(device)

# Defining the optimizer
MnistModel_optimizer_Raternn = torch.optim.Adam(Raternn.parameters(), lr=0.0002)

### loading the image generator ###

# Structure of the generator model we intend to import, this must stay constant too
G = nn.Sequential(
    nn.Linear(latent_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, image_size),
    nn.Tanh())
G.to(device)

# String is the path for the dictionary containing the generator weights we want to load in
G.load_state_dict(torch.load("DigitGen300Epochs.pt"))

# Eval modifies the initialization after loading the parameters somehow
# We write it as the alternative is training, which is certainly not what we are after here
G.eval()


### Functions concerning ELO rating system ###

def create_lopsided_rated_tensors(batch_size):
    from torchvision.transforms import ToTensor
    from torchvision.datasets import MNIST
    from random import randint

    Pref_rating = float(10)
    No_pref_rating = float(2)

    test_dataset = MNIST(root='data/',
                         train=False,
                         transform=ToTensor())

    RatedTensors = []

    for num in range(0, batch_size):

        selected_element = randint(0, len(test_dataset)-1)

        img, label = test_dataset[selected_element]

        if label == 6:

            RatedTensor = ((img), Pref_rating)
            RatedTensors.append(RatedTensor)

        #elif label == 7:

            #RatedTensor = ((img), Pref_rating)
            #RatedTensors.append(RatedTensor)

        #elif label == 9:

            #RatedTensor = ((img), Pref_rating)
            #RatedTensors.append(RatedTensor)

        else:

            RatedTensor = ((img), No_pref_rating)
            RatedTensors.append(RatedTensor)

    return RatedTensors


def create_initial_rated_tensors(batch_size, initialrating):


    RatedTensors = []

    for num in range(0, batch_size):

        # First, create a random input for Generator
        G_input_vector = torch.randn(latent_size).to(device)

        # Get the corresponding output
        tensor = G(G_input_vector)

        # Add rating and combine it with the tensor
        RatedTensor = ((tensor), initialrating)

        # Add the rating,tensor pair to the RatedTensor list
        RatedTensors.append(RatedTensor)

    return RatedTensors

def image_from_tensor(TensorA, TensorB):

    ImageGPU = TensorA.reshape(1, 28, 28)
    ImageDetatched = ImageGPU.cpu().detach()
    ImageFinalA = ToPILImage()(ImageDetatched)

    ImageGPU = TensorB.reshape(1, 28, 28)
    ImageDetatched = ImageGPU.cpu().detach()
    ImageFinalB = ToPILImage()(ImageDetatched)

    # ImageFinal.show()
    return ImageFinalA, ImageFinalB

def GUI(imageA, imageB):
    global choice
    root = Tk()

    my_imgA = ImageTk.PhotoImage(imageA)
    image_label = Label(image=my_imgA)
    image_label.grid(row=1, column=0)

    my_imgB = ImageTk.PhotoImage(imageB)
    image_label = Label(image=my_imgB)
    image_label.grid(row=1, column=2)

    def choseA():
        global choice
        choice = "a"
        root.destroy()

    def choseIndifferent():
        global choice
        choice = "e"
        root.destroy()

    def choseB():
        global choice
        choice = "b"
        root.destroy()

    def quit():
        global choice
        choice = "d"
        root.destroy()

    buttonA = Button(root, text="I prefer this image", padx=10, pady=0, command=choseA)
    buttonA.grid(row=2, column=0)

    buttonIndifferent = Button(root, text="They are equally preferable", padx=10, pady=0, command=choseIndifferent)
    buttonIndifferent.grid(row=2, column=1)

    buttonB = Button(root, text="I prefer this image", padx=10, pady=0, command=choseB)
    buttonB.grid(row=2, column=2)

    buttonquit = Button(root, text="Return to program", command=quit)
    buttonquit.grid(row=0, column=2)

    root.mainloop()
    return choice


def ELO_update_rating(choice, ratingA, ratingB):

    # First calculate expected score
    Ea = 1 / (1 + 10 ** ((ratingB - ratingA) / 400))
    Eb = 1 / (1 + 10 ** ((ratingA - ratingB) / 400))
    print(Ea, Eb)
    print(choice)


    # Then, depending on users choice, assign actual score
    if choice == "a":
        Sa = 1
        Sb = 0

    if choice == "b":
        Sa = 0
        Sb = 1

    if choice == "e":
        Sa = 0.5
        Sb = 0.5


    # Finally define k constant and update ratings based on expected and actual score
    # K is the ELO amount which is exchanged
    k = 4
    ratingA = ratingA + k * (Sa - Ea)
    ratingB = ratingB + k * (Sb - Eb)

    return ratingA, ratingB

def single_cycle(Tensors_with_rating):
    global cycle_num
    cycle_num = cycle_num + 1
    # Hopefully this will add another image decided by the decider to the list of rated tensors every 5 user inputs
    if cycle_num % 5 == 0:
        D_input_vectors = torch.randn(image_size).to(device)
        G_input_vectors = Decidernn(D_input_vectors)
        tensor = G(G_input_vectors)

        global RatedTensors
        global initialrating
        tensor.to(device)
        RatedTensor = ((tensor), initialrating)
        #print(RatedTensor)

        RatedTensors.append(RatedTensor)
        print("new tensor appended!")
        print("len ratedtensors: ",len(RatedTensors))

        single_cycle(RatedTensors)

    else:

        # Pull out two rated tensors and split the rating from the tensor itself
        tensorAnum = random.randint(0, batch_size - 1)
        tensorBnum = random.randint(0, batch_size - 1)

        print(tensorAnum, tensorBnum)
        len(RatedTensors)
        tensorA, ratingA = RatedTensors[tensorAnum]
        tensorB, ratingB = RatedTensors[tensorBnum]

        # Create images from the tensors and feed these into the GUI, then allow the user to make a choice
        imageA, imageB = image_from_tensor(tensorA, tensorB)
        choice = GUI(imageA, imageB)
        print(choice)

        if choice == "d":
            tensors, ratings = map(list, zip(*RatedTensors))
            print(ratings)
            print("all done")

            # While finished == False nn training continues. By changing it globally to True here, we end that training.
            global finished
            finished = True

            return ratings
        else:
            # Update the ELO ratings of the two tensors according to the choice above
            ratingA, ratingB = ELO_update_rating(choice, ratingA, ratingB)

            # Last thing to do is to put together RatedTensors[A] and RatedTensors[B] again, and make the whole process loop
            RatedTensors[tensorAnum] = tensorA, ratingA
            RatedTensors[tensorBnum] = tensorB, ratingB

            single_cycle(RatedTensors)

def single_cyclea(RatedTensors):

    # Pull out two rated tensors and split the rating from the tensor itself
    tensorAnum = random.randint(0, batch_size - 1)
    tensorBnum = random.randint(0, batch_size - 1)

    print(tensorAnum, tensorBnum)
    len(RatedTensors)
    tensorA, ratingA = RatedTensors[tensorAnum]
    tensorB, ratingB = RatedTensors[tensorBnum]

    # Create images from the tensors and feed these into the GUI, then allow the user to make a choice
    imageA, imageB = image_from_tensor(tensorA, tensorB)
    choice = GUI(imageA, imageB)
    print(choice)

    if choice == "d":
        tensors, ratings = map(list, zip(*RatedTensors))
        print(ratings)
        print("all done")

        # While finished == False nn training continues. By changing it globally to True here, we end that training.
        global finished
        finished = True

        return ratings
    else:
        # Update the ELO ratings of the two tensors according to the choice above
        ratingA, ratingB = ELO_update_rating(choice, ratingA, ratingB)

        # Last thing to do is to put together RatedTensors[A] and RatedTensors[B] again, and make the whole process loop
        RatedTensors[tensorAnum] = tensorA, ratingA
        RatedTensors[tensorBnum] = tensorB, ratingB

        single_cycle(RatedTensors)






    return

def percentile_promt(ELOratings):
    choice_two = input(
        "input r to get the percentile of a given rating, input p to get the rating of a given percentile, input anything else to end program.")
    if choice_two == "r":
        chosenRating = float(
            input("please input a rating number and you will get which percentile it is in in return:"))
        print(stats.percentileofscore(ELOratings, chosenRating))
    if choice_two == "p":
        chosenPercentile = float(
            input("Please input a percentile and you will get the rating which lies in that percentile:"))
        print(np.percentile(ELOratings, chosenPercentile))
    else:
        return


def train_Raternn(images, labels):

    # Model predictions given the input images
    outputs = Raternn(images)
    outputs = outputs.squeeze()
    outputs = outputs.double()

    # Cross entropy is not the right loss function here, because the labels(ratings) will have values which exceed the
    # number of distinct output nodes which the model possesses.

    # Mean square error (MSE) works though, because it produces loss according to the distance between the output and the label,
    # perfect for getting an algorithm to award the correct rating

    # Take note that for mse_loss to work the inputs need to be 64-bit floats, referred to as dtype-double by python
    # Also, .backward must be set to retain_graph=True, should try to understand what that does.
    # There is a more complicated, alternate fix which may provide significantly better performance
    loss = F.mse_loss(outputs, labels)

    # Reset gradients
    MnistModel_optimizer_Raternn.zero_grad()

    # Compute gradients
    loss.backward(retain_graph=True)

    # Adjust the parameters using backprop
    MnistModel_optimizer_Raternn.step()
    print("rater loss: ", loss.item())

    return loss, outputs, labels

def train_Decidernn():

    # Decider generates an image by feeding its output to the generator an output given a random input.
    # The decider itself takes a random input and, much like the generator in a GAN, seeks to produce a desirable output
    # from this random input.

    # making a random input
    D_input_vectors = torch.randn(image_size).to(device)

    # Creating an output from the random input
    G_input_vectors = Decidernn(D_input_vectors)

    # feeding the output to the Generator and receiving an image.
    image = G(G_input_vectors)

    # The image receives a rating from Raternn
    image_rating = Raternn(image)

    #outputs = outputs.squeeze()
    #outputs = outputs.double()

    # This is a super wierd loss function, but the idea is that loss will go down as rating goes up
    loss = 1/(1 + image_rating)

    # Reset gradients
    MnistModel_optimizer_Decidernn.zero_grad()
    print("decider loss: ", loss.item())
    # Compute gradients
    loss.backward(retain_graph=True)

    # Adjust the parameters using backprop
    MnistModel_optimizer_Decidernn.step()

    return loss


def training_loop(total_epochs, modeltrainer):
    #print("train loader len: ", len(train_loader))
    #print("rater")
    completed_epochs = 0
    train_loader = DataLoader(RatedTensors, len(RatedTensors), shuffle=True)
    print("len ratedtensors: ", len(RatedTensors))
    for epoch in range(total_epochs):
        completed_epochs = completed_epochs + 1
        for (images, labels) in (train_loader):
            # Load a batch, transform to vectors and send to gpu
            print(images)
            images = images.reshape(len(RatedTensors), -1).to(device)
            labels = labels.to(device)
            print("Passed critical failure point!")
            # Run the training loop
            modeltrainer(images, labels)





### ACTUAL PROGRAM ###

# The final incarnation of this programme will have three parts:
# 1. The initialization step, where everything which must be done before the program begins happens
# 2. The human loop, which is the part of the program which the human interacts with
# 3. The training loop, which runs in the background always, once initialization is done

# Each of these three part must be a single function for the threading to work properly

### Initialization ###

# Initialize by creating a bunch of tensors and assign them all a rating of 10
#RatedTensors = create_lopsided_rated_tensors(batch_size)

global RatedTensors
RatedTensors = create_initial_rated_tensors(batch_size, initialrating)
#print(RatedTensors)
print("len RatedTensors:", len(RatedTensors))

# First we define the training set as the set of already rated tensors
print("len rated tensors", len(RatedTensors))
train_ds = RatedTensors
# Then we define a dataloader so we can pass the rated tensors in batches (don't know if this is strictly neccessary)
train_loader = DataLoader(RatedTensors, len(RatedTensors), shuffle=True)  # , num_workers=4, pin_memory=True)

# This is how the training ends, if abort becomes something else than 0
finished = False

cycle_num = 0




### Human loop ###

# The human loop consists of images being pulled from RatedTensors or created by the Generator and then being presented
# to the human by the GUI. The human makes a choice, ELO is updated and the loop begins anew.

# apperenty threading expects args to be iterable, so if you pass it a list it has to be in the form of a single
# element tuple (or else it thinks every element of the list is a separate argument)
t1 = threading.Thread(target=single_cycle, args=(RatedTensors,))


## NN training ###

def NN_training():

    global finished

    # When the user chooses option d (for Done) in the GUI, finished is globally set to True and the while loop is broken
    while finished == False:
        training_loop(1, train_Raternn)

        train_Decidernn()

    else:
        return

t2 = threading.Thread(target=NN_training)

def sample_images():
    # making a random input
    D_input_vectors = torch.randn(image_size).to(device)

    # Creating an output from the random input
    G_input_vectors = Decidernn(D_input_vectors)

    # feeding the output to the Generator and receiving an image.
    image = G(G_input_vectors)

    image = image.cpu()

    image = image.detach().numpy()

    image = image.reshape(1, 28, 28)

    print(image.shape)

    plt.imshow(image[0], cmap='gray')
    plt.show()

t3=threading.Thread(target=sample_images)

t1.start()
t2.start()
#t3.start()
print("done")