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

from utils import *


def train_model(model, criterion, optimizer, num_epochs, train_dataloader, test_dataloader, early_stopping, model_path, device = 'cpu'):
    losses = []
    losses_test = []
    psnr_train = []
    psnr_test = []
    for epoch in tqdm(range(num_epochs)):
        loss_train = 0.0
        for inputs, labels in train_dataloader:
            optimizer.zero_grad()
            outputs = model(inputs.to(device))  ### change model
            # print(outputs.shape)
            # print(labels.shape)
            # print(inputs.shape)
            loss = criterion(outputs.to(device), labels.to(device))
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
            
        metric_train = PSNR_metric(outputs.to(device),labels.to(device), device).item()
        losses.append(loss_train/len(train_dataloader))
        psnr_train.append(metric_train)
        
        loss_test = 0.0
        model.eval() ### change model
        with torch.no_grad():
            for inputs, labels in test_dataloader:
                # print(outputs.shape)
                # print(labels.shape)
                # print(inputs.shape)
                outputs = model(inputs.to(device)) ### change model
                loss_t = criterion(outputs.to(device), labels.to(device))
                loss_test += loss_t.item()
                
        metric_test = PSNR_metric(outputs.to(device),labels.to(device), device).item()
        losses_test.append(loss_test/len(test_dataloader))
        psnr_test.append(metric_test)
        
        if epoch % 5 == 0:
            print(f"Epoch: {epoch}, Train loss: {round(losses[-1], 4)}, Train PSNR: {round(metric_train, 4)}, Test loss: {round(losses_test[-1],4)}, Test PSNR: {round(metric_test, 4)}")

        # Apply early stopping
        if early_stopping.step(losses[-1], model, model_path):
            print("Early stopping triggered. Stopping training.")
            break
            
        if epoch == num_epochs:
            torch.save(model.state_dict(), model_path)
        
    return losses, psnr_train, losses_test, psnr_test

def evaluate_model(model, train_dataloader,  device = 'cpu'):
    model.eval()
    with torch.no_grad():
        # Get a batch of data
        dataiter = iter(train_dataloader)
        low_res, high_res = next(dataiter)
        
        output = model(low_res.to(device))
        
    return output, high_res, low_res