import torch
import random
from torchvision.transforms import ToPILImage
import torchvision
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import numpy as np
from tkinter import *
from PIL import ImageTk,Image

# Defining device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Defining model parameters, these must stay constant or things will break due to the model we are loading
# requiring these values to work
image_size = 784
hidden_size = 256
latent_size = 64

# Global variables:
Initial_upper_bound_list = []
Initial_lower_bound_list = []
Initial_inputA = []
Initial_inputB = []


# Can be changed
batch_size = 2


# Structure of the model, this must stay constant too
G = nn.Sequential(
    nn.Linear(latent_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, image_size),
    nn.Tanh())
G.to(device);

# String is the path for the location of the dictionary containing the parameters we want to load
G.load_state_dict(torch.load("DigitGen300Epochs.pt"))

# Eval modifies the initialization after loading the parameters somehow
# We write it, as the alternative is training, which is certainly not what we are doing here
G.eval()


# Input vectors defined
# The main goal of this method is to hone in on the part of the input space which contains the images which the user is
# interested. This is a very important part, the inputs chosen must be bounded according to the previous choices by the user.

def first_bound_update(choice, hyperplane, upper_bound_list, lower_bound_list, InputA, InputB):
    element_num = 0
    # As these are the global bounds for the torch.randn function, they were the bounds for when the digit GAN was made.
    # This means that they are the bounds within which we can expect inputs to result in images.

    global_upper_bound = 3
    global_lower_bound = -3
    if choice == "a":
        try:
            for element in hyperplane:
                element_num = len(upper_bound_list)
                if element > InputA[element_num]:
                    upper_bound_list.append(element + 0.7)
                    lower_bound_list.append(global_lower_bound)
                if element < InputA[element_num]:
                    lower_bound_list.append(element - 0.7)
                    upper_bound_list.append(global_upper_bound)
                if element == InputA[element_num]:
                    lower_bound_list.append(element - 0.7)
                    upper_bound_list.append(element + 0.7)
        except IndexError:
                print(upper_bound_list)
                print(lower_bound_list)
                print(len(upper_bound_list))
                print(len(lower_bound_list))
                print("error")

    if choice == "b":
        try:
            for element in hyperplane:
                element_num = len(upper_bound_list)
                if element > InputB[element_num]:
                    upper_bound_list.append(element + 0.7)
                    lower_bound_list.append(global_lower_bound)
                if element < InputB[element_num]:
                    lower_bound_list.append(element - 0.7)
                    upper_bound_list.append(global_upper_bound)
                if element == InputB[element_num]:
                    lower_bound_list.append(element - 0.7)
                    upper_bound_list.append(element + 0.7)
        except IndexError:
                print(upper_bound_list)
                print(lower_bound_list)
                print(len(upper_bound_list))
                print(len(lower_bound_list))
                print("error")

# This takes the list of the maximum bounds made above and updates them based on chosen inputs

# By changing the

#                       if element > InputA[element_num]:
#                       upper_bound_list[element_num] = element
# To something like:    upper_bound_list[element_num] = upper_bound_list[element_num] + 0.5

# It would probably soothe the problem of the algorithm getting stuck on a single point after few iterations
# The problem is that this is not a solution - eventually the bounds will narrow and there will be no way to iterate
# Also, because the bounds will be narrowing much slower, it means that the program will seem much dumber to the user,
# straying from a good solution to a bad one for no obvious reason, as it randomly selects an input somewhere else in the
# input space. Since it is non - obvious to the user how adjacent different regions of the input space are, the hyperplane
# solution also introduces the possibility of ending up in the wrong area all together while trying to reach the desired
# digit.

# I implemented this change in a crude fashion and it confirms the worry. Trying to drive the images in a direction is
# agonizingly slow, and there is no way as a human to know if you are actually progressing in the right way.
# It feels like you are trying to heard the program along a path you yourself cannot see - which is true, in this
# implementation the human does all the work.

def later_bound_update(choice, upper_bound_list, lower_bound_list, InputA, InputB):
    element_num = 0
    hyperplane = []
    # This populates the set above with midpoints between the two inputs
    for element in InputA:
        midpoint = (InputA[element_num] + InputB[element_num])/2
        hyperplane.append(midpoint)
        element_num = element_num + 1

    if choice == "e":
        element_num = 0
        for element in lower_bound_list:
            lower_bound_list[element_num] = lower_bound_list[element_num] - 0.7
            element_num = element_num + 1
        element_num = 0
        for element in upper_bound_list:
            upper_bound_list[element_num] = upper_bound_list[element_num] + 0.7
            element_num = element_num + 1

    if choice == "a":
        element_num = 0
        for element in hyperplane:
            if element > InputA[element_num]:
                upper_bound_list[element_num] = element + 0.7
            if element < InputA[element_num]:
                lower_bound_list[element_num] = element - 0.7
            if element == InputA[element_num]:
                upper_bound_list[element_num] = element + 0.7
                lower_bound_list[element_num] = element - 0.7
            element_num = element_num + 1


    if choice == "b":
        element_num = 0
        for element in hyperplane:
            if element > InputB[element_num]:
                upper_bound_list[element_num] = element + 0.7
            if element < InputB[element_num]:
                lower_bound_list[element_num] = element - 0.7
            if element == InputB[element_num]:
                upper_bound_list[element_num] = element + 0.7
                lower_bound_list[element_num] = element - 0.7
            element_num = element_num + 1

# This function creates random inputs for the image generator from the space within the defined upper and lower bounds
def create_two_bounded_input(upper_bound_list, lower_bound_list):
    inputC = []
    inputD= []
    for element in upper_bound_list:
        element_num = len(inputC)
        random_number = random.uniform(lower_bound_list[element_num], upper_bound_list[element_num])
        inputC.append(random_number)
    for element in upper_bound_list:
        element_num = len(inputD)
        random_number = random.uniform(lower_bound_list[element_num], upper_bound_list[element_num])
        inputD.append(random_number)
    inputC = torch.Tensor(inputC)
    inputD = torch.Tensor(inputD)

    # return is very important for calling functions
    return inputC, inputD


### GUI ###

# Main problem currently is twofold. (1) the code which provides the user with information is separated from the
# code which handles user input. I must draw them together for the GUI to work. Some code lies in the
# initialization function, some in the iterative cycle function, and some in the image_from_input function below
# (2) is that I cant find a way to translate a tensor to a png without saving the png. I can build the
# GUI around saving and accessing relevant images, but this is messy. I would rather dive into the translation library
# and hopefully cut out the part that saves the png, leaving only the translation, but this might be hard.

# Turns out the solution to (2) is just to use torchvision to convert a tensor to a PIL image

def image_from_input(InputA, InputB):
    Input = torch.Tensor(InputA).to(device)
    ImageRaw = G(Input)
    ImageGPU = ImageRaw.reshape(1, 28, 28)
    ImageDetatched = ImageGPU.cpu().detach()
    ImageFinalA = ToPILImage()(ImageDetatched)

    Input = torch.Tensor(InputB).to(device)
    ImageRaw = G(Input)
    ImageGPU = ImageRaw.reshape(1, 28, 28)
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

    buttonNeither = Button(root, text="They are equally preferable", padx=10, pady=0, command=choseIndifferent)
    buttonNeither.grid(row=2, column=1)

    buttonB = Button(root, text="I prefer this image", padx=10, pady=0, command=choseB)
    buttonB.grid(row=2, column=2)

    buttonquit = Button(root, text="Return to program", command=quit)
    buttonquit.grid(row=0, column=2)

    root.mainloop()
    return choice

def boundaries_to_choice(upper_bound_list, lower_bound_list):
    inputA, inputB = create_two_bounded_input(upper_bound_list,lower_bound_list)
    imageA, imageB = image_from_input(inputA, inputB)
    choice = GUI(imageA, imageB)
    return choice

def initialization():

    # Initialize by creating two random tensors
    InputA = torch.randn(latent_size)
    InputB = torch.randn(latent_size)

    # This creates a hyperplane out of the midpoints of each corresponding element in the two input vectors
    hyperplane = (InputA + InputB) / 2

    print(hyperplane)
    print(len(hyperplane))


    # This makes images out of the input tensors for the user to evaluate #
    imageA, imageB = image_from_input(InputA, InputB)

    #The GUI presents the images and the user makes a choice
    choice = GUI(imageA, imageB)

    # We have an issue here where the program fails if choice != a or b, but I can' be bothered to fix it
    # Below is a half-finished fix
    #print(choice)
    #while choice not in ["a","b"]:
    #    print("please make a choice")
    #    initialization()
    #else:
    #    return


    # This creates the empty lists which will be used to narrow the bounds within which we select future random inputs
    lower_bound_list = []
    upper_bound_list = []

    first_bound_update(choice, hyperplane, upper_bound_list, lower_bound_list, InputA, InputB)

    print(upper_bound_list)
    upper_bound_list = torch.Tensor(upper_bound_list)
    lower_bound_list = torch.Tensor(lower_bound_list)
    print(upper_bound_list)
    inputC, inputD = create_two_bounded_input(upper_bound_list, lower_bound_list)

    # Should probably just return these values and input them into the loop function, but i guess this works too
    # Bit unsure now what is really going on here, but it seems im just using global variables instead
    global Initial_upper_bound_list
    global Initial_lower_bound_list
    global Initial_inputA
    global Initial_inputB
    Initial_upper_bound_list = upper_bound_list
    Initial_lower_bound_list = lower_bound_list
    Initial_inputA = inputC
    Initial_inputB = inputD




# Once initialization hes been completed, the iterative cycle goes like this:
# make and show images from inputs >>> user makes choice in the GUI >>> update bounds based on choice >>>
# create new inputs based on new bounds >>> make and show images from inputs
# This cycle can be broken by user by inputting "d" at the choice stage
def iterative_cycle(upper_bound_list, lower_bound_list, InputA, InputB):
    print(InputA)
    print(InputB)

    imageA, imageB = image_from_input(InputA, InputB)
    choice = GUI(imageA, imageB)
    print(choice)
    if choice == "d":
        print("All done!")
        return

    if choice == "p":
        print("upper_bound_list:", upper_bound_list)

        try:
            print("hyperplane:", hyperplane)
        except NameError:
            print("lower_bound_list:", lower_bound_list)
            iterative_cycle(upper_bound_list, lower_bound_list, InputA, InputB)

        print("lower_bound_list:", lower_bound_list)
        iterative_cycle(upper_bound_list, lower_bound_list, InputA, InputB)

    later_bound_update(choice, upper_bound_list, lower_bound_list, InputA, InputB)

    # This is how calling a function should be done
    InputA, InputB = create_two_bounded_input(upper_bound_list, lower_bound_list)

    iterative_cycle(upper_bound_list, lower_bound_list, InputA, InputB)



initialization()
iterative_cycle(Initial_upper_bound_list,Initial_lower_bound_list,Initial_inputA,Initial_inputB)



#print(random_number)
#print("InputC:")
#print(InputC)
#print(upper_bound_list)
#print(lower_bound_list)
#print("success")








