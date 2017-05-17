import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple
import os

import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
import torchvision.transforms as T
import pickle
import glob
import time
import subprocess
from collections import namedtuple
import resource
import math





for i in range(100):
    yhat = Variable(torch.zeros(step, 1))
    target = Variable(torch.zeros(step, 1))
    target[-1, 0] = 1
    cx = Variable(torch.zeros(1, 2))
    hx = Variable(torch.zeros(1, 2))
    hidden = [hx, cx]

    for j in range(step):
        x = Variable(torch.zeros(1, 5))
        if j is 0:
            x += 1
        y, hx, cx = model(x, hidden)
        print(hx.data.numpy())
        hidden = (hx, cx)
        yhat[j] = y.clone()

    print('done - Hoping the last value should be zero')

    # learning
    optimizer.zero_grad()
    error = ((yhat - target) * (yhat - target)).mean()
    error.backward()
    optimizer.step()