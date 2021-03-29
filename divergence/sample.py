import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import divergence.data as dd
import divergence.models as dm
import divergence.training as dt

import torch
from torchvision import transforms
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset


if __name__ == "__main__":
    
    # define hyper-parameters
    train_batch = 64
    test_batch = 1000
    learning_rate = 1e-2
    momentum = 0.5
    l2_penalty = 1e-3
    training_epochs = 2
    validation_split = 0.1
    
    # define file path to the numpy binary
    npz_dump = "/content/gdrive/My Drive/acse4_4_data.npz"
    
    # transform to normalized tensor
    test_transf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.25)  # TODO: compute actual mu and sigma
    ])
    
    # transform with data augmentation for training
    train_transf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        test_transf
    ])
    
    # load train data with data augmentation transforms applied
    train_set = dd.ACSE44Dataset(npz_dump, transform=train_transf, train=True)
    
    # split train data into train and validation set (90-10 split)
    train_idx, validation_idx = dd.split_dataset(train_set, validation_split)
    
    # define validation set as a subset of the trainig set; don't do data augmentation here
    validation_set = Subset(dd.ACSE44Dataset(npz_dump, transform=test_transf, train=True), validation_idx)
    
    # load test set, again without data augmentation
    test_set = dd.ACSE44Dataset(npz_dump, transform=test_transf, train=False)
    
    # define data loaders
    train_loader = DataLoader(
        train_set,
        sampler=SubsetRandomSampler(train_idx),  # randomly sample from train_idx only
        batch_size=train_batch, drop_last=True
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=test_batch,
        drop_last=False, shuffle=False
    )
    test_loader = DataLoader(
        test_set,
        batch_size=test_batch,
        drop_last=False, shuffle=False
    )
    
    # instantiate model, optimizer and loss function
    model = dm.LeNet5()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=l2_penalty)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # train model
    trained_model, loss, accuracy, f1_score = dt.train_model(model, optimizer, loss_fn, n_epochs=training_epochs,
                                                             train_loader=train_loader,
                                                             validation_loader=validation_loader,
                                                             device='cuda')
    
    # make predictions on test set
    predictions = dt.make_predictions([trained_model], test_loader, "/content/gdrive/My Drive/sample_predictions.csv")
    
    print(predictions)
    