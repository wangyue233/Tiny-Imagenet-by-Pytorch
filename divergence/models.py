from enum import Enum

from torch import nn
from torchvision import models


##########################################################################################
class LeNet5(nn.Module):
    """ LeNet5, adapted for 3x64x64 input and 200 classes output """
    
    ### ####################################
    def __init__(self):
        super(LeNet5, self).__init__()
        self.c1 = nn.Conv2d(3, 16, kernel_size=11)
        self.s2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.c3 = nn.Conv2d(16, 64, kernel_size=5)
        self.s4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc = nn.Linear(7744, 200)
        
        self.act = nn.ReLU()
    
    ### ####################################
    def forward(self, x):
        x = self.act(self.c1(x))
        x = self.s2(x)
        x = self.act(self.c3(x)) 
        x = self.s4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


##########################################################################################
class VGG(nn.Module):
    """ CNN constructure inspired from https://juejin.im/entry/5bf51d35e51d454049668d57
    
    Delete several final layers from the origin structure of VGG to enable faster training.
    """

    ### ####################################
    def __init__(self):
        super(VGG,self).__init__()
        self.conv1 = nn.Conv2d(3,64,3,padding=1)
        self.conv2 = nn.Conv2d(64,64,3,padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        self.conv3 = nn.Conv2d(64,128,3,padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3,padding=1)
        self.pool2 = nn.MaxPool2d(2, 2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128,128, 3,padding=1)
        self.conv6 = nn.Conv2d(128, 128, 3,padding=1)
        self.conv7 = nn.Conv2d(128, 128, 1,padding=1)
        self.pool3 = nn.MaxPool2d(2, 2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.fc14 = nn.Linear(128*10*10,1024)
        self.drop1 = nn.Dropout2d()
        # self.fc15 = nn.Linear(1024,1024)
        # self.drop2 = nn.Dropout2d()
        self.fc16 = nn.Linear(1024,200)

    ### ####################################
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.relu(x)


        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        x = x.view(-1, 128*10*10)
        x = self.relu(self.fc14(x))
        x = self.drop1(x)
        # x = self.relu(self.fc15(x))
        # x = self.drop2(x)
        x = self.fc16(x)

        return x


##########################################################################################
def GoogLeNet_transfer_classifier(state_dict=None):
    """ GoogLeNet for transfer learing
    with the last FC layer reset and the other parameters fix
    
    Parameters
    ----------
    state_dict : OrderedDict, optional
        trained model parameters
        
    Returns
    -------
    googlenet : torch.nn.Module
        the network
    
    trainable_parameters : generator
        pass these parameters to the optimizer
    """
    googlenet = models.googlenet(pretrained=True, aux_logits=False)
    for param in googlenet.parameters():
        param.requires_grad = False
    googlenet.fc = nn.Linear(googlenet.fc.in_features, 200)
    
    if state_dict is not None:
        googlenet.load_state_dict(state_dict)
    
    trainable_parameters = googlenet.fc.parameters()
    
    return googlenet, trainable_parameters


##########################################################################################
class FineTuningModel(Enum):
    resnet18 = "resnet18"
    resnet34 = "resnet34"
    resnet50 = "resnet50"
    wide_resnet50 = "wide_resnet50"
    wide_resnet50_bn = "wide_resnet50_bn"
    vgg16 = "vgg16"
    googlenet = "googlenet"


##########################################################################################
def FineTuningClassifier(model, dropout_rate=0.5, state_dict=None):
    """ Load model with pre-trained weights for fine-tuning.
    Resets last FC layer with 200 outputs and adds a dropout layer before that.
    
    Loads parameters from state_dict if given. Otherwise returns with pre-trained
    weights from ImageNet, except for the last layer.
    
    Parameters
    ----------
    model : str
        which model to use
    
    dropout_rate : float, optional
        parameter for added dropout layer
    
    state_dict : OrderedDict, optional
        trained model parameters
    
    Returns
    -------
    resnet : torch.nn.Module
        the network
    """
    model = FineTuningModel(model)
    
    if model == FineTuningModel.resnet18:
        network = models.resnet18(pretrained=True)
        last_layer_inputs = network.fc.in_features
        network.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(last_layer_inputs, 200)
        )
        
    elif model == FineTuningModel.resnet34:
        network = models.resnet34(pretrained=True)
        last_layer_inputs = network.fc.in_features
        network.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(last_layer_inputs, 200)
        )
        
    elif model == FineTuningModel.resnet50:
        network = models.resnet50(pretrained=True)
        last_layer_inputs = network.fc.in_features
        network.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(last_layer_inputs, 200)
        )
    
    elif model == FineTuningModel.wide_resnet50:
        network = models.wide_resnet50_2(pretrained=True)
        last_layer_inputs = network.fc.in_features
        network.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(last_layer_inputs, 200)
        )
    
    elif model == FineTuningModel.wide_resnet50_bn:
        network = models.wide_resnet50_2(pretrained=True)
        last_layer_inputs = network.fc.in_features
        network.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(last_layer_inputs, 200),
            nn.BatchNorm1d(200)
        )
        
    elif model == FineTuningModel.vgg16:
        network = models.vgg16(pretrained=True)
        network.classifier[6].out_features = 200
    
    elif model == FineTuningModel.googlenet:
        network = models.googlenet(pretrained=True, aux_logits=False)
        network.fc = nn.Linear(network.fc.in_features, 200)
        
    else:
        raise NotImplementedError("importing '{}' is not implemented".format(model.value))
    
    if state_dict is not None:
        network.load_state_dict(state_dict)
    
    return network
