import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from utils_q4 import Attention,LogisticRegression,GRU,RNN,data_generator,MyDataset
np.random.seed(42)
torch.manual_seed(42)


# visualize the samples

dir = "./results/"
os.makedirs(dir,exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')


hidden_size = # write your code here

batch_size = # write your code here
epochs = # write your code here
lr = # write your code here


x,y = data_generator(0.02)
plt.figure(figsize=(15, 5))
plt.scatter(x,y)
plt.grid()
plt.title('samples from sine function with Gaussian noises')
plt.savefig(dir + "samples.png")
plt.cla()

seq_len = 10
x_slices = []
y_slices = []

# write your code here for creating the dataset 

train_ds = # write your code here
train_loader = # write your code here

def train(train_loader, device, model, criterion, optimizer, epoch):
    # write your code here

# RNN

rnn = # write your code here
print(rnn)
optim_rnn = # write your code here
criterion = # write your code here
rnn_history = []  

for epoch in tqdm(range(epochs)):
    rnn_loss = train(train_loader, device, rnn, criterion, optim_rnn, epoch)
    rnn_history.append(rnn_loss)

print(f'RNN loss = {rnn_loss}')
plt.plot(np.arange(epochs), np.array(rnn_history))
plt.title('Running loss history with RNN')
plt.grid()
plt.savefig(dir + "rnn.png")
plt.cla()

# GRU
gru = # write your code here
print(gru)
optim_gru = # write your code here
criterion = # write your code here
gru_history = []

for epoch in tqdm(range(epochs)):
    gru_loss = train(train_loader, device, gru, criterion, optim_gru, epoch)
    gru_history.append(gru_loss)

print(f'GRU loss = {gru_loss}')
plt.plot(np.arange(epochs), np.array(gru_history))
plt.title('Running loss history with GRU')
plt.grid()
plt.savefig(dir + "gru.png")
plt.cla()

attention = # write your code here
print(attention)
optim_attention = # write your code here
criterion = # write your code here
attention_history = []

for epoch in tqdm(range(epochs)):
    attention_loss = train(train_loader, device, attention, criterion, optim_attention, epoch)
    attention_history.append(attention_loss)
    
print(f'Attention loss = {attention_loss}')
plt.plot(np.arange(epochs), np.array(attention_history))
plt.title('Running loss history with Attention')
plt.grid()
plt.savefig(dir + "attention.png")
plt.cla()


logistic = # write your code here
print(logistic)
optim_logistic = # write your code here
criterion = # write your code here
logistic_history = []

for epoch in tqdm(range(epochs)):
    logistic_loss = train(train_loader, device, logistic, criterion, optim_logistic, epoch)
    logistic_history.append(logistic_loss)
    
print(f'Logistic regression loss = {logistic_loss}')
plt.plot(np.arange(epochs), np.array(logistic_history))
plt.title('Running loss history with logistic regression')
plt.grid()
plt.savefig(dir + "lr.png")
plt.cla()

def predict(model, title):
    
    # write your code here
    
    plt.plot(np.arange(0, 200, 1), preds)
    plt.grid()
    plt.title(f'prediction results using {title}')
    plt.savefig(dir + "pred_" + title + ".png")
    plt.cla()   
     
start = time.time()
predict(rnn, 'RNN')
end = time.time()
print(f'{end - start:.4f}s')

start = time.time()
predict(gru, 'GRU')
end = time.time()
print(f'{end - start:.4f}s')

start = time.time()
predict(attention, 'Attention')
end = time.time()
print(f'{end - start:.4f}s')

start = time.time()
predict(logistic, 'logistic regression')
end = time.time()
print(f'{end - start:.4f}s')