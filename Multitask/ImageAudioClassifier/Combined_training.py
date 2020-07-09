#!/usr/bin/env python
# coding: utf-8

# In[1]:

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import torch.utils.data as data
from yolonanoutils import *
from yolonano import *

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from mobilenet import *
from sklearn import decomposition
from sklearn import manifold
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

import copy
import random
from Helper import *
from utilsCombined import *
import time
import os
import sys


# In[2]:


SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


# Setup Image Classification network

# In[3]:


# vgg11_layers = get_vgg_layers(vgg11_config, batch_norm = True)

OUTPUT_DIM = 10

# CombinedModel = VGG(vgg11_layers, 10,30)
CombinedModel = mobilenetv2()
# CombinedModel = YOLONano()

print(CombinedModel)


pretrained_size = 224
pretrained_means = [0.485, 0.456, 0.406]
pretrained_stds= [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
                           transforms.Resize(pretrained_size),
                           transforms.RandomRotation(5),
                           transforms.RandomHorizontalFlip(0.5),
                           transforms.RandomCrop(pretrained_size, padding = 10),
                           transforms.ToTensor(),
                           transforms.Normalize(mean = pretrained_means, 
                                                std = pretrained_stds)
                       ])

test_transforms = transforms.Compose([
                           transforms.Resize(pretrained_size),
                           transforms.ToTensor(),
                           transforms.Normalize(mean = pretrained_means, 
                                                std = pretrained_stds)
                       ])

ROOT = '.data'

train_data = datasets.CIFAR10(ROOT, 
                              train = True, 
                              download = True, 
                              transform = train_transforms)

test_data = datasets.CIFAR10(ROOT, 
                             train = False, 
                             download = True, 
                             transform = test_transforms)

VALID_RATIO = 0.9

n_train_examples = int(len(train_data) * VALID_RATIO)
n_valid_examples = len(train_data) - n_train_examples

print('No of training examples', n_train_examples)
print('\n')
print('No of valid examples', n_valid_examples)

train_data, valid_data = data.random_split(train_data, 
                                           [n_train_examples, n_valid_examples])

valid_data = copy.deepcopy(valid_data)
valid_data.dataset.transform = test_transforms

BATCH_SIZE = 32

train_iterator = data.DataLoader(train_data, 
                                 shuffle = True, 
                                 batch_size = BATCH_SIZE)

valid_iterator = data.DataLoader(valid_data, 
                                 batch_size = BATCH_SIZE)

test_iterator = data.DataLoader(test_data, 
                                batch_size = BATCH_SIZE)


# In[4]:


START_LR = 1e-7

# params = [
#           {'params': CombinedModel.features.parameters(), 'lr': START_LR / 10},
#           {'params': CombinedModel.classifier1a.parameters()},
#            {'params': CombinedModel.classifier1b.parameters()},
#          ]

params = [
          {'params': CombinedModel.features.parameters(), 'lr': START_LR / 10},
          {'params': CombinedModel.classifier1a.parameters()},
          {'params': CombinedModel.classifier1b.parameters()},

         ]

optimizer = optim.Adam(params, lr = START_LR)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

criterion = nn.CrossEntropyLoss()

CombinedModel = CombinedModel.to(device)
criterion = criterion.to(device)

FOUND_LR = 0.001



optimizerIC = optim.Adam(CombinedModel.parameters(), lr = FOUND_LR)


# Setup Audio Classifier

# In[5]:


train_path = 'gcommand_toy_example/train'
test_path = 'gcommand_toy_example/test'
valid_path = 'gcommand_toy_example/valid'

window_size = 0.02
window_stride= 0.01
window_type = 'hamming'
max_len = 101

# loading data
train_dataset = ClassificationLoader(train_path, 0.02, 0.01,
                                     'hamming', True, 101, augment=True)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=32, shuffle=True,
    num_workers=20, pin_memory=True, sampler=None)

valid_dataset = ClassificationLoader(valid_path,  0.02, 0.01,
                                     'hamming',True, 101)

valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=None,
                                           num_workers=20, pin_memory=True, sampler=None)

test_dataset = ClassificationLoader(test_path,  0.02, 0.01,
                                     'hamming', True, 101)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=32, shuffle=None,
    num_workers=20, pin_memory=True, sampler=None)


# In[6]:


# CombinedModel = torch.nn.DataParallel(CombinedModel).cuda()

# params = [
#           {'params': CombinedModel.features.parameters(), 'lr': 0.0005},
#           {'params': CombinedModel.classifier2a.parameters()},
#           {'params': CombinedModel.classifier2b.parameters()},

#          ]

params = [
          {'params': CombinedModel.features.parameters(), 'lr': START_LR / 10},
          {'params': CombinedModel.classifier2a.parameters()},
          {'params': CombinedModel.classifier2b.parameters()},

         ]

optimizerAudio = optim.Adam(CombinedModel.parameters(), lr = 0.0005)


# In[7]:

# 
# optimizer = optim.Adam(CombinedModel.parameters(), lr = 0.001)


# In[ ]:


best_valid_loss = np.inf
iteration = 0
epoch = 1
best_valid_lossIC = float('inf')

# trainint with early stopping
ctr=0
while (epoch < 100 + 1) and (iteration < 10):
    valid_loss, acc = Audiotest(valid_loader, CombinedModel, True)
    if (acc<80 or ctr>=4):
        ctr=0
        print('----Training Audio classifier------')
        Audiotrain(train_loader, CombinedModel, optimizerAudio, epoch, True, 100)
        valid_loss, acc = Audiotest(valid_loader, CombinedModel, True)
        if valid_loss > best_valid_loss:
            iteration += 1
            print('Loss was not improved, iteration {0}'.format(str(iteration)))
        else:
            print('Saving model...')
            iteration = 0
            best_valid_loss = valid_loss
#             state = {
#                 'net':  CombinedModel.state_dict() if True else CombinedModel.state_dict(),
#                 'acc': valid_loss,
#                 'epoch': epoch,
#                 'class_num': 30
#             }
#             if not os.path.isdir('gcommand_pretraining_model/'):
#                 os.mkdir('gcommand_pretraining_model/')
#             torch.save(state, 'gcommand_pretraining_model/' + '/' + 'test2')
        epoch += 1
    
    else:
        ctr+=1
        print('Audio seems fine.. moving to image training:',ctr)

        
    print('---Performance on Image Classifier after audio training------')
    valid_lossIC, valid_accIC = evaluate_IC(CombinedModel, valid_iterator, criterion, device)
    print(f'\t Val. Loss: {valid_lossIC:.3f} |  Val. Acc: {valid_accIC*100:.2f}%')
    
    state = {
                'net':  CombinedModel.state_dict() if True else CombinedModel.state_dict(),
                'acc': valid_loss,
                'epoch': epoch,
                'class_num': 30
            }
    if not os.path.isdir('gcommand_pretraining_model/exp4/'):
        os.mkdir('gcommand_pretraining_model/exp4/')
    torch.save(state, 'gcommand_pretraining_model' + '/' + 'exp4/'+'audioside-{0:.2f}_{1:.2f}'.format(valid_accIC*100,acc))
    
    start_time = time.time()
    print('---------Image Classification Training --------')
    train_lossIC, train_accIC = train_IC(CombinedModel, train_iterator, optimizerIC, criterion, device)
    valid_lossIC, valid_accIC = evaluate_IC(CombinedModel, valid_iterator, criterion, device)
        
    if valid_lossIC < best_valid_lossIC:
        best_valid_losICs = valid_lossIC
#         torch.save(CombinedModel.state_dict(), 'Imageside.pt')

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_lossIC:.3f} | Train Acc: {train_accIC*100:.2f}%')
    print(f'\t Val. Loss: {valid_lossIC:.3f} |  Val. Acc: {valid_accIC*100:.2f}%')
    
    print('--------Audio classification Performance after image training----')
    valid_loss, acc = Audiotest(valid_loader, CombinedModel, True)
    
    
    state = {
        'net':  CombinedModel.state_dict() if True else CombinedModel.state_dict(),
        'acc': valid_loss,
        'epoch': epoch,
        'class_num': 30
    }
    print('Saving model...')
    torch.save(state, 'gcommand_pretraining_model' + '/' + 'exp4/'+'imageside-{0:.2f}_{1:.2f}'.format(valid_accIC*100,acc))
    
# test model
test(test_loader, CombinedModel, True)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




