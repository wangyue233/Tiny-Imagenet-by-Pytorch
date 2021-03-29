import divergence.training as dt

import os
from enum import Enum

import torch
from torch import nn
from torch.nn.functional import softmax
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from xgboost import XGBClassifier


##########################################################################################
class CombinationMethod(Enum):
    hard_voting = "hard_voting"
    soft_voting = "soft_voting"
    logistic_regression = "logistic_regression"
    xgboost = "xgboost"


##########################################################################################
def combine_predictions(probabilities, filenames, out_file=None, method='majority_vote',
                        weights=None, train_probabilities=None, train_labels=None,
                        device='cpu'):
    """ Combines probabilities from different models into one classification prediction
    Returns pandas.DataFrame with predictions, optionally saves to csv file.
    
    Methods
    -------
    majority_vote: 
        class with highest probabililty among majority of models is chosen
        
    average_probability:
        probabilities are averaged across models (optional: weighted average)
    
    xgboost:
        Trains XGBoost classifier on per-model predictions on trainig data.
        
    logistic_regression:
        A logistic regression classifier is trained on trainig data.
    
    Parameters
    ----------
    probabilities : numpy.ndarray
        probabilities calculated by the different models
        axis: [sample, class, model]
    
    filenames : array-like
        list of file names corresponding to the test images (in lower case)
    
    out_file : Path-like, optional
        csv file path for predictions
        
    method : str, optional
        one of the methods above
    
    weights : array-like, optional
        weight given to predictions of the models
        
    train_probabilities : numpy.ndarray, optional
        probabilities calculated by the different models on the training set
        axis = [sample, class, model]
    
    train_labels : numpy.ndarray, optional
        training set labels
    
    device : str, optional
        'cpu' or 'cuda', hardware accelerator to use
    
    Returns
    -------
    frame : pandas.DataFrame
        columns = [Filename, Label]
        data = predicted labels
        
    """
    method = CombinationMethod(method)        
    
    if method == CombinationMethod.soft_voting:
        predictions = soft_voting(probabilities, weights)
    
    elif method == CombinationMethod.hard_voting:
        predictions_per_model = np.argmax(probabilities, axis=1)
        predictions = hard_voting(predictions_per_model)
    
    elif method == CombinationMethod.logistic_regression:
        assert train_probabilities is not None and train_labels is not None,\
            "for 'logistic_regression' method, train_probabilities and train_labels have to be provided"
        
        predictions = logistic_regression(probabilities, train_probabilities, train_labels, device)
    
    elif method == CombinationMethod.xgboost:
        assert train_probabilities is not None and train_labels is not None,\
            "for 'xgboost' method, train_probabilities and train_labels have to be provided"
            
        predictions_per_model = np.argmax(probabilities, axis=1)
        train_predictions = np.argmax(train_probabilities, axis=1)
        
        predictions = xgboost(predictions_per_model, train_predictions, train_labels)
        
    else:
        raise NotImplementedError("Method '{}' not implemented".format(method.value))

    ##
    frame = pd.DataFrame({"Filename": filenames, "Label": predictions})
    
    if out_file is not None:
        if os.path.exists(out_file):
            os.remove(out_file)
        frame.to_csv(out_file, sep=',', index=False)
    
    return frame


##########################################################################################
def hard_voting(per_model_predictions, weights=None):
    """ Combines predictions of multiple models by majority vote
    
    Arguments
    ---------
    per_model_predictions : numpy.ndarray
        axis = [sample, model]; data = predicted class (int)
    
    weights : numpy.ndarray(dtype=int), optional
        weights to give to models
    
    Returns
    -------
    predictions : numpy.ndarray
        axis = [sample]; data = combined predicted class (int)
        
    """
    if weights is not None:
        assert len(weights) == per_model_predictions.shape[1],\
            "list of weights must have same length as number of different models"
        weights = np.array(weights, dtype=int)
    else:
        weights = np.ones(per_model_predictions.shape[1], dtype=int)
    
    predictions = list()
    for i in range(per_model_predictions.shape[0]):
        counts = np.bincount(per_model_predictions[i], weights)
        if (counts == counts.max()).sum() != 1:
            print("Warning: no majority for sample {:d}".format(i))
        predictions.append(np.argmax(counts))
    
    return np.array(predictions)
    

##########################################################################################
def soft_voting(probabilities, weights=None):
    """ Combines predictions of multiple models by majority vote
    
    Arguments
    ---------
    probabilities : numpy.ndarray
        axis = [sample, class, model]
        data = probability calculated by each model for this class
    
    weights : numpy.ndarray, optional
        weights to give to models
    
    Returns
    -------
    predictions : numpy.ndarray
        axis = [sample]; data = combined predicted class (int)
        
    """
    
    if weights is not None:
        assert len(weights) == probabilities.shape[2],\
            "list of weights must have same length as number of different models"
        for i in range(probabilities.shape[2]):
            probabilities[..., i] *= weights[i]
    
    average = probabilities.sum(axis=2)
    predictions = np.argmax(average, axis=1)
    
    return predictions
    

##########################################################################################
def xgboost(per_model_predictions, train_predictions, train_labels):
    """ Trains XGBoost classifier on per-model predictions on a training set,
    then makes a combined prediction on the test set
    
    Arguments
    ---------
    probabilities : : numpy.ndarray
        labels predicted by the different models on the test set
        axis = [sample, class, model]
    
    train_probabilities : numpy.ndarray
        labels predicted by the different models on the training set
        axis = [sample, class, model]
    
    train_labels : numpy.ndarray
        training set true labels
        
    Returns
    -------
    predictions : numpy.ndarray
        axis = [sample]; data = combined predicted class (int)
    
    """
    xgbmodel = XGBClassifier(learning_rate=0.01)
    xgbmodel.fit(train_predictions, train_labels)
    
    predictions = xgbmodel.predict(per_model_predictions)
    
    return predictions
    
    
##########################################################################################
def logistic_regression(probabilities, train_probabilities, train_labels, device='cpu'):
    """ Trains a logistic regression classifier using pytorch's SGD
    
    Arguments
    ---------
    probabilities : : numpy.ndarray
        probabilities calculated by the different models on the test set
        axis = [sample, class, model]
    
    train_probabilities : numpy.ndarray
        probabilities calculated by the different models on the training set
        axis = [sample, class, model]
    
    train_labels : numpy.ndarray
        training set labels
        
    Returns
    -------
    predictions : numpy.ndarray
        axis = [sample]; data = combined predicted class (int)
    """
    
    in_features = train_probabilities.shape[1] * train_probabilities.shape[2]
    
    tensor_set = TensorDataset(
        torch.from_numpy(train_probabilities.astype(np.float32).reshape(train_probabilities.shape[0], in_features)),
        train_labels
    )
    
    train_loader = DataLoader(
        tensor_set, batch_size=128, drop_last=True, shuffle=True
    )
    
    classifier = nn.Linear(in_features, 200)
    optimizer = torch.optim.SGD(classifier.parameters(), lr=4e-3, momentum=0.2)
    loss_fn = nn.CrossEntropyLoss()
    
    trained_classifier = dt.train_model(
        classifier, optimizer, loss_fn, n_epochs=11, train_loader=train_loader, device=device
    )[0]
    
    trained_classifier.eval()
    predicted_probs = softmax(trained_classifier(torch.from_numpy(probabilities.astype(np.float32).reshape(probabilities.shape[0], in_features)).to(device)), dim=-1)
    predicted_classes = predicted_probs.max(1)[1].detach().cpu().numpy()
    
    return predicted_classes
