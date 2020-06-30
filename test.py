from glob import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
from torchvision import transforms
from torchvision import models
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch import optim
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader


# plot dog photos from the dogs vs cats dataset
from matplotlib import pyplot
from matplotlib.image import imread



# define location of dataset
catfolder = './data/Cats/'
# plot first few images
for i in range(6):
	# define subplot
	pyplot.subplot(330 + 1 + i)
	# define filename
	filename = catfolder + 'cat' + str(i) + '.jpg'
	# load image pixels
	image = imread(filename)
	
	# plot raw pixel data
	pyplot.imshow(image,)
# show the figure
pyplot.show()
'''
dogFolder = './data/dogs/'
for i in range(6):
	pyplot.subplot(330+1+i)
	filename = dogFolder+ 'dog'+str(i)+'.jpg'
	image = imread(filename)
	pyplot.imshow(image)

pyplot.show()


'''