import os
import shutil
import random
from enum import Enum

import torch
from torch.nn.functional import softmax
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from livelossplot import PlotLosses

TPU_AVAILABLE = bool(os.environ.get('COLAB_TPU_ADDR'))
if TPU_AVAILABLE:
    import torch_xla
    import torch_xla.core.xla_model as xm
    from torch_xla.distributed.parallel_loader import ParallelLoader as xla_loader


##########################################################################################
def set_seed(seed):
    """ Use this to set ALL the random seeds to a fixed value
    and take out any randomness from cuda kernels.
    
    Parameters
    ----------
    seed (int)
    
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return True


##########################################################################################
def train_model(model, optimizer, criterion, n_epochs, train_loader,
                validation_loader=None, device='cpu', random_seed=42,
                backup_folder=None):
    """ Train a model for a number of epochs.
    Visualizes average loss, F1 and accuracy score over epochs
    If a folder is given, saves state dict and scores to disk after each epoch

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model to train
    
    optimizer : torch.optim.Optimizer
        Optimizer for to use in training
    
    criterion : loss function in torch.nn
    
    n_epochs : int
        number of training iterations over entire train dataset
    
    train_loader : torch.utils.data.DataLoader
        batch data loader of train data
    
    validation_loader : torch.utils.data.DataLoader, optional
        batch data loader of validation data, if available
        
    device : str, optional
        'cpu' or 'cuda'; hardware accelerator to use

    random_seed : int, optional
        seed number for RNGs
    
    backup_folder : path-like, optional
        folder where model parameters are saved after each epoch
        !will delete all contents of folder first!
    
    Returns
    -------
    model : torch.nn.Module
        model with optimized weights after training
    
    validation_loss : float
        average loss of final model on all samples from validation/test set
    
    validation_accuracy : float
        accuracy score of final model on validation/test set
    """
    
    set_seed(random_seed)  # seed all RNGs before start to have reproducible results
    model = model.to(device)  # create instance of model
    
    if backup_folder is not None:
        if os.path.isdir(backup_folder):
            shutil.rmtree(backup_folder)
        elif os.path.exists(backup_folder):
            os.remove(backup_folder)
        
        os.mkdir(backup_folder)
        with open(os.path.join(backup_folder, "training_report.csv"), 'w') as csv_report:
            csv_report.write("epoch,train_loss,train_accuracy,train_f1,validation_loss,validation_accuracy,validation_f1\n")

    # use special (faster) data loaders if running on TPU
    if TPU_AVAILABLE and (device not in ['cpu', 'cuda']):
        train_loader = xla_loader(train_loader, [device]).per_device_loader(device)
        if validation_loader is not None:
            validation_loader = xla_loader(validation_loader, [device]).per_device_loader(device)

    live_plot = PlotLosses()
    # training loop
    for epoch in range(n_epochs):
        logs = dict()
        # do mini-batch SGD over all training samples
        train_loss, train_accuracy, train_f1 = train(model, optimizer, criterion, train_loader, device)
        
        if backup_folder is not None:
            torch.save(model.state_dict(), os.path.join(backup_folder, "model_epoch_{:d}.pth".format(epoch)))

        logs['log loss'] = train_loss
        logs['f1 score'] = train_f1
        logs['accuracy'] = train_accuracy

        # evaluate model on validation/test set
        if validation_loader is not None:
            validation_loss, validation_accuracy, validation_f1 = validate(model, criterion, validation_loader, device)
            logs['val_log loss'] = validation_loss
            logs['val_f1 score'] = validation_f1
            logs['val_accuracy'] = validation_accuracy
        else:
            validation_loss, validation_accuracy, validation_f1 = 0, 0, 0

        if backup_folder is not None:
            with open(os.path.join(backup_folder, "training_report.csv"), 'a') as csv_report:
                csv_report.write("{:d},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n".format(
                    epoch, train_loss, train_accuracy, train_f1, validation_loss, validation_accuracy, validation_f1
                ))
        
        # draw the visualization of average loss and accuracy
        live_plot.update(logs)
        live_plot.draw()
      
    return model, validation_loss, validation_accuracy, validation_f1


##########################################################################################
def train(model, optimizer, criterion, data_loader, device='cpu'):
    """ trains model for one epoch

    Params
    ------
    model : torch.nn.Module
        The neural network model to train
    
    optimizer : torch.optim.Optimizer
        Optimizer for to use in training
    
    criterion : loss function in torch.nn

    data_loader : torch.utils.data.Dataset or torch.utils.data.DataLoader
        Batch data loader of training data
    
    device : str, optional
        'cpu' or 'cuda'; hardware accelerator to use

    Returns
    -------
    average training loss (float)

    average trainig accuracy score (float)
    
    average training F1 score (float)

    """
    model.train()  # set model into training mode
    train_loss, train_accuracy, train_f1 = 0, 0, 0
    num_samples = 0  # count total number of train images to return average loss and accuracy score at the end

    for X, y in data_loader:
        X, y = X.to(device), y.to(device)       # send data to memory of hardware accelerator
        optimizer.zero_grad()                   # set all parameter gradients to 0
        
        predictions = model(X)                  # run model on batch
        loss = criterion(predictions, y)        # calcualte loss
        loss.backward()                         # calculate gradients
        train_loss += loss.item() * X.size(0)
        
        y_pred = softmax(predictions, dim=-1).max(1)[1]          # calculate predictions
        train_accuracy += accuracy_score(y.cpu().numpy(), y_pred.detach().cpu().numpy()) * X.size(0)
        train_f1 += f1_score(y.cpu().numpy(), y_pred.detach().cpu().numpy(), average='weighted') * X.size(0)
        num_samples += X.size(0)
        
        # make optimizer step (needs special function if running on TPU)
        if TPU_AVAILABLE and (device not in ['cpu', 'cuda']):
            xm.optimizer_step(optimizer)
        else:
            optimizer.step()

    return train_loss/num_samples, train_accuracy/num_samples, train_f1/num_samples


##########################################################################################
def validate(model, criterion, data_loader, device='cpu'):
    """ evaluates the model accuracy on data from data_loader

    Parameters
    ------
    model : torch.nn.Module
        The neural network model to evaluate
    
    criterion : loss function in torch.nn

    data_loader : torch.utils.data.Dataset or torch.utils.data.DataLoader
        Batch data loader of test data
    
    device : str, optional
        'cpu' or 'cuda'; hardware accelerator to use

    Returns
    -------
    average loss per sample (float)

    accuracy score (float)
    
    F1 score (float)

    """
    model.eval()  # set model into evaluation mode (e.g. dropout filters not active, L2 loss not included in loss)
    validation_loss, validation_accuracy, validation_f1 = 0, 0, 0
    num_samples = 0

    for X, y in data_loader:
        with torch.no_grad():       # don't waste time on calculating gradients
            X, y = X.to(device), y.to(device)
            predictions = model(X)
            loss = criterion(predictions, y)
            validation_loss += loss.item() * X.size(0)
            num_samples += X.size(0)
            y_pred = softmax(predictions, dim=-1).max(1)[1]
            validation_accuracy += accuracy_score(y.cpu().numpy(), y_pred.cpu().numpy()) * X.size(0)
            validation_f1 += f1_score(y.cpu().numpy(), y_pred.detach().cpu().numpy(), average='weighted') * X.size(0)
            
    return validation_loss/num_samples, validation_accuracy/num_samples, validation_f1/num_samples


##########################################################################################
def predict(model, data_loader, device='cpu'):
    """ Generates array of probabilities calculated by a model
    
    Parameters
    ----------
    model : torch.nn.Module
        The trained neural network
        must return output from last layer without softmax activation
    
    data_loader : torch.utils.data.DataLoader
        batch data loader
        
    device : str, optional
        'cpu' or 'cuda'; hardware accelerator to use
    
    Returns
    -------
    probabilities : numpy.ndarray
        axis = [sample, class]
    
    """
    model = model.to(device)
    model.eval()
        
    # use special (faster) data loaders if running on TPU
    if TPU_AVAILABLE and (device not in ['cpu', 'cuda']):
        data_loader = xla_loader(data_loader, [device]).per_device_loader(device)
    
    probabilities = list()
    
    for X, _ in data_loader:
        with torch.no_grad():
            X = X.to(device)
            probs = softmax(model(X), dim=-1)
            probabilities.append(probs)
    
    probabilities = torch.cat(probabilities, dim=0)
    
    return probabilities.detach().cpu().numpy()
