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
import time

# from sklearn.metrics import confusion_matrix

from torchvision import datasets, transforms

import itertools

gound_truth_list = []
answer_list = []

total_epoch = 100
Leaning_Rate = 0.001

# �� �̸�
model_type = "mymodel"


# https://matplotlib.org/api/_as_gen/matplotlib.pyplot.imshow.html
# pyplot�� imshow�� Ȱ���� ��
def imshow(inp, cmap=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp, cmap)


class Net(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(500, 50, bias=True)
        self.fc2 = nn.Linear(50, 9)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # print("x shape1:{}".format(x.shape))

        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # print("x shape2:{}".format(x.shape))

        x = x.view(x.size(0), -1)
        # print("x shape3:{}".format(x.shape))

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return F.log_softmax(x, -1)


def fit(epoch, model, data_loader, phase='training', volatile=False):
    """epoch = n_try, model = network, data_loader = image data files, phase = step"""

    optimizer = optim.SGD(model.parameters(), lr=Leaning_Rate, momentum=0.5)

    if phase == 'training':
        model.train()
    if phase == 'validation':
        model.eval()
        volatile = True

    running_loss = 0.0
    running_correct = 0

    for batch_idx, (data, target) in enumerate(data_loader):
        """Loop data_loader. batch_idx =  """
        if phase=='training':
            optimizer.zero_grad()
        output = model(data)

        loss = F.nll_loss(output,target)
        running_loss +=F.nll_loss(output,target, size_average= False).data

        preds = output.data.max(dim=1, keepdim=True)[1]

        gound_truth = target.data

        # print("preds:{}".format(preds))

        answer = preds.squeeze()

        # print("gound_truth:{}".format(gound_truth))
        # print("answer:{}".format(answer))

        a = gound_truth.data.detach().cpu().numpy()
        b = answer.data.detach().cpu().numpy()

        gound_truth_list.append(a)
        answer_list.append(b)

        # print("ground_truth numpy:{}".format(a))
        # print("answer numpy:{}".format(b))

        running_correct += preds.eq(target.data.view_as(preds)).cpu().sum()

        if phase == 'training':
            loss.backward()
            optimizer.step()

         loss = running_loss / len(data_loader.dataset)
        accuracy = 100. * running_correct.item() / len(data_loader.dataset)
        print(
        f'{phase} loss is {loss:{5}.{2}} and {phase} accuracy is {running_correct}/{len(data_loader.dataset)}{accuracy:{10}.{4}}')

        # print("gound_truth_list:{}".format(gound_truth_list))
        # print("answer_list:{}".format(answer_list))

        return loss, accuracy