import torch
import torchvision
from torchvision.transforms import ToTensor, Normalize, Compose, ToPILImage
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import random


Tensor_list = []
Initial_rating = 1000

#Defining device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#Defining model parameters, these must stay constant as these are the dimensions of the model we are loading in
image_size = 784
hidden_size = 256
latent_size = 64

# Can be changed
batch_size = 1

# Structure of the Generator model, this must stay constant too
G = nn.Sequential(
    nn.Linear(latent_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, image_size),
    nn.Tanh())
G.to(device);
G.load_state_dict(torch.load("DigitGen300Epochs.pt"))
# String is the path of the file with the parameters

# Eval modifies the initialization after loading the parameters somehow
# We write it as the alternative is training, which is certainly not what we are after here
G.eval()


input_vectors = torch.randn(batch_size, latent_size).to(device)
fake_images_raw = G(input_vectors)
print(fake_images_raw)


# The Decider takes a random input and passes the output onto the Generator, it thereby decides which region of the
# input space will be used by the generator to make an image
# The Rater rates the image  generated and the resultant rating is used to train the decider (higher rating = lower loss)
# Loss function works by making a probability density function of human ratings and loss being given by the percentile
# which the rating of the generated image falls within. (So if the rating is in the 20th percentile loss = 0.2 3rd percentile = 0.03)
Decider = nn.Sequential(
    nn.Linear(image_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, latent_size),
    nn.Tanh())
Decider.to(device)
Decider.eval()

# input is big to give variation(?) output is latent_size, as this is the size of Generator input
# Could also feed decider with images, but that is wierd (or...)


# The Rater trains by assessing itself according to the ratings of the images which the human has viewed where loss is
# defined by how close the guessed rating is to the true (human) rating of a given image
# True ratings are determined by images being compared by the human and receiving or losing rating according to the
# ELO rating system
# The human-rated images are placed into a list which serves as the training set for the Rater
# The loss function for the rater will have to be something like a sigmoidal or logarithmic function with limit of y=1
# based off how far away it guesses from the true rating.
Rater = nn.Sequential(
    nn.Linear(image_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, latent_size),
    nn.ReLU(),
    nn.Linear(latent_size, 1),
    nn.Tanh())
Rater.train()

# Input is image_size, since it will be fed images. Output size is 1 as we want a single number as our rating
# There might be some kind of upper limit to the values the nn can generate and this could create issues for the
# Elo rating process as it may artificially limit the sensitivity of the Rater. (ex. if a really good image is generated
# it may only give it a mediocre rating as it is unable to produce a value high enough to reflect the goodness of the
# image)


# This creates an image by passing the decider output to the generator

def Decide_tensor():
    decider_input_vector = torch.randn(image_size).to(device)
    generator_input_vector = Decider(decider_input_vector)
    decided_tensor = G(generator_input_vector)
    return decided_tensor

def New_tensor_to_list(Tensor_list, initial_rating):
    tensor = Decide_tensor()
    RatedTensor = ((tensor), initial_rating)
    Tensor_list.append(RatedTensor)

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


    # Finally define k constand and update ratings based on expected and actual score
    k = 120
    ratingA = ratingA + k * (Sa - Ea)
    ratingB = ratingB + k * (Sb - Eb)

    return ratingA, ratingB

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

def Cycle_2(Tensor_list):
    # Pull out two rated tensors and split the rating from the tensor itself
    tensorAnum = random.randint(0, len(Tensor_list) - 1)
    tensorBnum = random.randint(0, len(Tensor_list) - 1)

    tensorA, ratingA = Tensor_list[tensorAnum]
    tensorB, ratingB = Tensor_list[tensorBnum]

    # Create images from the tensors and feed these into the GUI, then allow the user to make a choice
    imageA, imageB = image_from_tensor(tensorA, tensorB)
    choice = GUI(imageA, imageB)
    print(choice)

    if choice == "d":
        tensors, ratings = map(list, zip(*Tensor_list))
        print(ratings)
        print("all done")
        return ratings
    else:
        # Update the ELO ratings of the two tensors according to the choice above
        new_ratingA, new_ratingB = ELO_update_rating(choice, ratingA, ratingB)

        # Last thing to do is to pair the tensors with their respective updated ratings
        Tensor_list[tensorAnum] = tensorA, new_ratingA
        Tensor_list[tensorBnum] = tensorB, new_ratingB

    return


tensor_
tensor = Decide_tensor()


ImageGPU = sample_image.reshape(1, 28, 28)
ImageDetatched = ImageGPU.cpu().detach()
ImageFinal = ToPILImage()(ImageDetatched)
ImageFinal.show()



rating = 1000
#print(input_vectors)

sample = G(input_vectors)





Tensors = []
Tensormod = (sample[0])
x = Tensormod[0]
Tensor = (x, rating)
Tensors.append(Tensor)
print(sample)
print(Tensors)
#save_fake_images(0)
