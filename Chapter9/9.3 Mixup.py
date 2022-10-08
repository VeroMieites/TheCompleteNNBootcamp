import torch
from PIL import Image
import os
from random import randint
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
#   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -
# mixup consists on adding a random (image * p), from the batch, with the original (image * (p-1))
# The label has to be a mix of two labels too:
# xhat = lambda * xi + (1-lambda) * xj -> data
# yhat = lambda * yi + (1-lambda) * yj -> label
#   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -
# Im gonna use MNIST as image dataset
import torchvision.datasets as datasets
from torchvision import transforms
# gotta download them the first time
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())

# get a batch size of 10
batch = 10
batch_x = mnist_trainset.data[:batch]
batch_y = mnist_trainset.targets[:batch]

# Some functions that would need to be used in general cases
def standarize_image(x):
    mean = np.mean(x,axis=(0,1))
    std = np.std(x,axis=(0,1))
    x = x/x.max()
    return (x-mean)/std, mean, std
def destandarize_image(x,mean,std):
    x = std*x+mean
    return np.clip(x,0,1)

# Mixup procedure: This would get done for every image in the batch.
# Start by getting the lambda and the random image to apply to the current image in mixup
lam = 0.5 # this value will make it harder or easier for the model to learn
batch_size = len(batch_x)
current_image,current_mean,current_std = standarize_image(np.array(batch_x[0]))
random_index = randint(0, batch_size-1)
random_image_from_batch,random_mean,random_std = standarize_image(np.array(batch_x[random_index]))

# some plots
plt.figure()
plt.imshow(destandarize_image(current_image,current_mean,current_std))
plt.axis('off')
plt.title('Destandarized current (true) image')

plt.figure()
plt.imshow(destandarize_image(random_image_from_batch,random_mean,random_std))
plt.axis('off')
plt.title('Destandarized random (to add) image')

# mix the images
mixed_image = lam * current_image + (1-lam) * random_image_from_batch
mixed_mean = lam * current_mean + (1-lam) * random_mean
mixed_std = lam * current_std + (1-lam) * random_std

plt.figure()
plt.imshow(destandarize_image(mixed_image,mixed_mean,mixed_std))
plt.axis('off')
plt.title(f'Destandarized mixed (lambda={lam}) image')

# We would have to redefine the loss too, according to both labels
# loss = lambda * crossE(pred,y_a) + (1-lambda) crossE(pred,y_b)
#Careful, the crossE requires one-hot vectors so, for four classes, the labels would be
# c1 = [1,0,0,0], c2 = [0,1,0,0], c3 = [0,0,1,0] and c4 = [0,0,0,1]
# crossE = -1/n sum_j^n sum_i^c (yi log(yhati)), with c == classes and n== items in the batch
# loss is only calculated for correct predictions
# this time, we would have intermediate classes made out of lambda*c1 + (1-lambda)*c3, for example.
