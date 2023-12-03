# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, TensorDataset, DataLoader
from tqdm.notebook import tqdm
from torchinfo import summary
import torch.nn.functional as F
import math

from torchmetrics.image import PeakSignalNoiseRatio

from torchvision import transforms
from PIL import Image

def set_device():
    device = 'cpu'
    if torch.cuda.device_count() > 0 and torch.cuda.is_available():
        print("Cuda installed! Running on GPU!")
        device = torch.device(torch.cuda.current_device())
        print(f'Device: {device} {torch.cuda.get_device_name(device)}')
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f'Device: {device} {(device)}')
    else:
        print("No GPU available!")
    return device

def set_seed(seed):
    """Set all random seeds to a fixed value and take out any randomness from cuda kernels
    """
    #random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled   = False

def im_file_to_tensor(image_path, image_file, res = 'low'):
    complete_path = image_path + str('/') + image_file
    image = Image.open(complete_path)
    grayscale_transform = transforms.Grayscale()
    transform = transforms.ToTensor()
    if res == 'low':
        tensor_image = transform(grayscale_transform(image))#.reshape(200,200,1)
    else:
        tensor_image = transform(grayscale_transform(image))#.reshape(800,800,1)
    return tensor_image

def stack_images(image_path, im_lst, res = 'low'):
    image_tensors = []
    for image_file in im_lst:
        tensor = im_file_to_tensor(image_path, image_file, res)
        image_tensors.append(tensor)
    return torch.stack(image_tensors, dim=0)

def PSNR_metric(super_res, high_res, device = 'cpu'):
    """ Compute the peak signal to noise ratio, measure quality of image"""
    psnr = PeakSignalNoiseRatio().to(device)
    return psnr(super_res, high_res)

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = np.inf

    def step(self, val_loss, model, model_path):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False





