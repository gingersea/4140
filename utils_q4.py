import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt
import time
from tqdm import tqdm

np.random.seed(42)
torch.manual_seed(42)
def data_generator(noise_parameter):
  # write your code here
  return x,y

class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, idx):
        return self.x[idx].float(), self.y[idx].float()

class RNN(nn.Module):
    
    # write your code here

    
class GRU(nn.Module):
    
    # write your code here
    
    
class Attention(nn.Module):
    
    # write your code here
    
class LogisticRegression(nn.Module):
    
    # write your code here
