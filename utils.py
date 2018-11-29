# Forked from https://github.com/ernest-s/fastai_notebooks/blob/master/utils.py

# Credit Ernest Kirubakaran

import os
import sys
import PIL
import math
import copy
import torch
import itertools
import torchvision
import numpy as np
import pandas as pd
from time import time
from PIL import Image
import torch.nn as nn
from torch._six import inf
import torch.optim as optim
from functools import partial
import torch.nn.functional as F
import matplotlib.pyplot as plt
from bisect import bisect_right
from torch.optim import Optimizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets, models, transforms

class lr_finder():
    
    """Implementation of LR Find function explained in the paper:
        https://arxiv.org/abs/1506.01186
        Tries a range of Learning Rates and returns the loss for
        the entire range.
        
        Args:
            model: Network for which the LR has to be found
            criterion: Loss function
            optimizer: pytorch optimizer
                Optimizer can have different param_groups for
                    different layers of the network to have 
                    different LRs. The layers should be in order
                    of closeness to the input. i.e. the layer 
                    closest to the input should be the first and
                    the last layer before the loss function should
                    be the last
            dataloaders: Dictionary containing DataLoader for "train"
            device: "CPU" or "CUDA"
            factor: Factor by which the LR has to be reduced for the
                earlier groups. For e.g. if the factor is 10, then if 
                the last group has a LR of 0.2, the group ahead of it 
                will have a LR of 0.02 and the one ahead of the second
                group will have an LR of 0.002 and so on.
            freeze_bn: Should batch norm layers be frozen? Set True while
                    using pre-trained models (big networks) for imagenet
                    like images
    """
    
    def __init__(self, model, criterion, optimizer, dataloaders, 
                 device, factor=10, freeze_bn = False):
        
        model = model.to(device)
        self.model = model.train()
        if freeze_bn == True:
            model.apply(set_bn_eval)
        self.criterion = criterion
        self.optimizer = optimizer
        self.dataloaders = dataloaders
        self.device = device
        self.results = None
        self.factor = factor
        
    def fit(self):
        
        # Count number of examples in train dataset
        train_examples = len(self.dataloaders['train'].dataset)
        train_bs = self.dataloaders['train'].batch_size
        # Count number of mini batches
        mini_batches = train_examples // train_bs
        increment = 16 / mini_batches
        
        # Start with a very low LR for the last layer
        lr_hist = []
        # Save loss history of the last layer
        loss_hist = []
        cur_lr = 1e-5
        
        mini_batch = 0
        
        model = self.model
        criterion = self.criterion
        optimizer = self.optimizer
        device = self.device
        for inputs, labels in self.dataloaders['train']:
            
            # Print status bar
            mini_batch_comp = int((mini_batch/mini_batches)*100)//2
            sys.stdout.write('\r')
            sys.stdout.write("[%-50s] %d%%" %("="*mini_batch_comp, 
                             2*mini_batch_comp))
            mini_batch += 1
            
            # Set differential learning rates for various param groups based 
            # on factor value
            fa = 0
            for pg in optimizer.param_groups[::-1]:
                pg['lr'] = cur_lr / (self.factor ** fa)
                fa += 1
            lr_ = []
            for pg in optimizer.param_groups:
                lr_.append(pg['lr'])
            lr_hist.append(lr_)
            optimizer.zero_grad()
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            loss_hist.append(loss.item())
            cur_lr = cur_lr + (lr_hist[-1][-1] * increment)
            
            if cur_lr == 0:
                cur_lr += increment
            
            # Stop iteration at a LR of 1e1
            if cur_lr > 10:
                break
                
            # Stop iteration is loss is getting too high
            if (loss > (5*loss_hist[0])) & (cur_lr > .1):
                break
            
            if (lr_hist[-1][-1] * 10 - cur_lr) < (increment * lr_hist[-1][-1]):
                # Reset when LR is reduced by a factor of 10 
                cur_lr = round(lr_hist[-1][-1]*10, 8)
        
        if len(lr_hist) > len(loss_hist):
            lr_hist.pop(-1)
                
        self.results = {'lr': lr_hist, 'loss': loss_hist}
        self.results['loss'] = self.moving_average()
                     
    def plot_lr(self):
        
        # Plot LR vs Loss
        if self.results:
            lr_ = [i[-1] for i in self.results["lr"]]
            plt.plot(lr_, self.results['loss'])
            plt.xscale("log")
            plt.show()
        else:
            print("Results not available")
            
    def moving_average(self):
        # Smoothen the loss by calculating moving average
        loss_hist = np.cumsum(self.results['loss'])
        loss_hist /= np.arange(1, len(loss_hist)+1)
        return loss_hist
    
    def lr_schedule(self, group = -1):
        # Plot LR schedule
        lr_ = [i[group] for i in self.results["lr"]]
        plt.plot(lr_)
        
class _LRScheduler_(object):
    # Modified code from torch.optim.lr_scheduler
    def __init__(self, optimizer, last_batch=-1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        if last_batch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming" 
                                   "an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], 
                                 optimizer.param_groups))
        self.step(last_batch + 1)
        self.last_batch = last_batch

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() 
                if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.
        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_lr(self):
        raise NotImplementedError

    def step(self, batch=None):
        if batch is None:
            batch = self.last_batch + 1
        self.last_batch = batch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
            
class CosineAnnealingLR_(_LRScheduler_):
    # Modified code from torch.optim.lr_scheduler
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:
    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 +
        \cos(\frac{T_{cur}}{T_{max}}\pi))
    When last_epoch=-1, sets initial lr as lr.
    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. Note that this 
    only implements the cosine annealing part of SGDR, and not the restarts.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.
    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer, T_max, eta_min=0, last_batch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        super(CosineAnnealingLR_, self).__init__(optimizer, last_batch)

    def get_lr(self):
        return [self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * self.last_batch / self.T_max)) / 2
                for base_lr in self.base_lrs]
        
class SGDR(CosineAnnealingLR_):
    r"""Implements cosine annealing of LR
    Args:
        iterations: no. of iterations after which LR has to be reset
            in general iterations must be equal to no. of minibatches in 
            a training epoch
        cycle_mult: factor by which the iteration cycle has to be 
            increased after every cycle. Default: 1
    """
    def __init__(self, optimizer, T_max, eta_min = 0, last_batch = -1, 
                 cycle_mult = 1):
        self.cycle_mult = cycle_mult
        super(SGDR, self).__init__(optimizer, T_max, eta_min, last_batch)
        
    def step(self, batch=None):
        if batch is None:
            batch = self.last_batch + 1
        self.last_batch = batch
        if ((self.last_batch%self.T_max) == 0) & (self.last_batch != 0):
            # Reset after T_max number of iterations are reached
            self.last_batch = 0
            self.T_max = self.T_max * self.cycle_mult
            # Increase T_max by cycle_mult after each cycle
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
class visualize_results:
    '''
    Helps visualizing validation images and predicions
    '''
    
    def __init__(self, preds, probs, val_y, img_names, classes, 
                 path, file_st = "path"):
        '''
        Args:
            preds: class predictions
            probs: class probabilities
            val_y: true class
            img_names: image names
            classes: class names
            path: path to image files
            file_st: file structure. 
                How the image files are organized 
                path: Different class images are in their respective folders
                csv: class information is in csv file
        '''
        self.preds = preds
        self.probs = probs
        self.val_y = val_y
        self.img_names = img_names
        self.classes = classes
        self.path = path
        self.file_st = file_st
    
    def rand_by_mask(self, mask): 
        return np.random.choice(np.where(mask)[0], min(len(self.preds), 4), 
                                replace=False)
    
    def rand_by_correct(self, is_correct): 
        mask = (self.preds == self.val_y) == is_correct
        return self.rand_by_mask(mask)
    
    def plots(self, ims, fig_size=(16,8), rows=1, titles=None):
        f = plt.figure(figsize=fig_size)
        for i in range(len(ims)):
            sp = f.add_subplot(rows, len(ims)//rows, i+1)
            sp.axis('Off')
            if titles is not None: 
                sp.set_title(titles[i], fontsize=16)
            plt.imshow(ims[i])
            
    def load_img_id(self, idx): 
        c = self.classes[self.val_y[idx]]
        i = self.img_names[idx]
        if self.file_st == "path":
            return np.array(PIL.Image.open(f'{self.path}{c}/{i}'))
        else:
            return np.array(PIL.Image.open(f'{self.path}{i}'))
    
    def plot_images(self, title, mode = "random", is_correct = True, 
                    cls = None, uncertain = False):
        '''
        Args:
            mode: "random" or "most"
                if "random", pick predicted records at random
                if "most", rank the predictions and pick the top
            iscorrect: 
                if True, pick the correct predictions
                if False, pick the wrong predictions
            uncertain: pick the most uncertain ones from a given class
            cls:
                The class which has to be picked
                Take integer, pick from the class corresponding to that
                    integer
        '''
        if mode == "random":
            idxs = self.rand_by_correct(is_correct)
        elif mode == "most":
            if cls is None:
                print("Please mention class for mode = 'most'")
                return
            elif uncertain == True:
                idxs = self.most_uncertain(cls)
            else:
                idxs = self.most_by_correct(cls, is_correct)
        imgs = [self.load_img_id(x) for x in idxs]
        actuals = [self.classes[self.val_y[x]] for x in idxs]
        predictions = [self.classes[self.preds[x]] for x in idxs]
        title_probs = [self.probs[x][self.val_y[x]] for x in idxs]
        title_probs = ['{0:.{1}f}'.format(i, 3) for i in title_probs]
        tp = [f'{i}/\n{j}/\n{k}' for i, j, k in zip(actuals, predictions, 
              title_probs)]
        print(title)
        print("Actuals/Predictions/Prob")
        if len(imgs) > 0:
            return self.plots(imgs, rows=1, titles=tp)
        else:
            print('Not Found.')
            
    def most_by_mask(self, mask, mult, y):
        idxs = np.where(mask)[0]
        return idxs[np.argsort(mult * self.probs[:,y][idxs])[:4]]
    
    def most_by_correct(self, y, is_correct): 
        return self.most_by_mask(((self.preds == self.val_y)==is_correct) 
                                 & (self.val_y == y), (-1)**(is_correct), y)
    
    def most_uncertain(self, cls):
        cls_prob = self.probs[:,cls]
        next_max_prob = np.max(np.delete(self.probs, cls, axis = 1), axis = 1)
        diff = np.argsort(np.abs(cls_prob - next_max_prob))
        return diff[(self.val_y == cls)[diff]][:4]
    
def set_bn_eval(m):
    # Function to freeze batch norm layers of huge networks
    # https://discuss.pytorch.org/t/freeze-batchnorm-layer-lead-to-nan/8385
    # ruotianluo Ruotian(RT) Luo
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
      m.eval()
