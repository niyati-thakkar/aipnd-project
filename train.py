import argparse
import pandas as pd
import numpy as np
import seaborn as sns

import torch
from torch import nn, tensor, optim
import torch.nn.functional as F

from torch.autograd import Variable
from torchvision import datasets, transforms

import torchvision.models as models

import argparse
from collections import OrderedDict

import json
import PIL
from PIL import Image
import time

from os.path import isdir

def arg_parser():
    parser = argparse.ArgumentParser(description="Train.py")
    parser.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str)
    parser.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
    parser.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
    parser.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=120)
    parser.add_argument('--epochs', dest="epochs", action="store", type=int, default=1)
    parser.add_argument('--gpu', dest="gpu", action="store", default="gpu")
    args = parser.parse_args()
    return args

# Function to create a transformer for training data
def train_transformer(train_dir):
    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    return train_data

# Function to create a transformer for test/validation data
def test_transformer(test_dir):
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    return test_data

# Function to create a data loader
def data_loader(data, train=True):
    if train:
        loader = torch.utils.data.DataLoader(data, batch_size=50, shuffle=True)
    else:
        loader = torch.utils.data.DataLoader(data, batch_size=50)
    return loader

# Function to check and set the device (CPU or GPU)
def check_gpu(gpu_arg):
    if not gpu_arg:
        return torch.device("cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        print("CUDA not found, using CPU instead.")
    return device

# Function to load a pre-trained model
def load_pretrained_model(architecture="vgg16"):
    model = models.vgg16(pretrained=True)
    model.name = "vgg16"
    for param in model.parameters():
        param.requires_grad = False
    return model

# Function to create an initial classifier for the model
def create_classifier(model, hidden_units):
    classifier = nn.Sequential(OrderedDict([
        ('inputs', nn.Linear(25088, 120)),
        ('relu1', nn.ReLU()),
        ('dropout', nn.Dropout(0.5)),
        ('hidden_layer1', nn.Linear(120, 90)),
        ('relu2', nn.ReLU()),
        ('hidden_layer2', nn.Linear(90, 70)),
        ('relu3', nn.ReLU()),
        ('hidden_layer3', nn.Linear(70, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    model.classifier = classifier
    return classifier

# Function to perform validation on the model
def validation(model, testloader, criterion, device):
    test_loss = 0
    accuracy = 0
    for ii, (inputs, labels) in enumerate(testloader):
        inputs, labels = inputs.to(device), labels.to(device)
        output = model.forward(inputs)
        test_loss += criterion(output, labels).item()
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    return test_loss, accuracy

# Function to train the neural network
def train_network(model, trainloader, validloader, device, criterion, optimizer, epochs, print_every, steps):
    if type(epochs) == type(None):
        epochs = 12
        print("Number of Epochs specified as 5.")

    print("Training process initializing...\n")

    for e in range(epochs):
        running_loss = 0
        model.train()

        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                with torch.no_grad():
                    valid_loss, accuracy = validation(model, validloader, criterion, device)

                print("Epoch: {}/{} | ".format(e+1, epochs),
                      "Training Loss: {:.4f} | ".format(running_loss/print_every),
                      "Validation Loss: {:.4f} | ".format(valid_loss/len(validloader)),
                      "Validation Accuracy: {:.4f}".format(accuracy/len(validloader)))

                running_loss = 0
                model.train()

    return model

# Function to validate the model on test data
def validate_model(model, testloader, device):
    correct, total = 0, 0
    with torch.no_grad():
        model.eval()
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy on test images is: %d%%' % (100 * correct / total))

# Function to save the model at a defined checkpoint
def save_checkpoint(model, save_dir, train_data):
    if type(save_dir) == type(None):
        print("Model checkpoint directory unknown, model not saved.")
    else:
        if isdir(save_dir):
            model.class_to_idx = train_data.class_to_idx
            torch.save({
                'structure': 'alexnet',
                'hidden_layer1': 120,
                'dropout': 0.5,
                'epochs': 12,
                'state_dict': model.state_dict(),
                'class_to_idx': model.class_to_idx,
                'optimizer_dict': optimizer.state_dict()
            }, 'checkpoint.pth')

            model.class_to_idx = train_data.class_to_idx
            checkpoint = {
                'architecture': model.name,
                'classifier': model.classifier,
                'class_to_idx': model.class_to_idx,
                'state_dict': model.state_dict()
            }

            torch.save(checkpoint, 'my_checkpoint.pth')
        else:
            print("Directory not found, model not saved.")

# Main function
def main():
    # Get Keyword Args for Training
    # Get Keyword Args for Training
    args = arg_parser()
    
    # Set directory for training
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Pass transforms in, then create trainloader
    train_data = test_transformer(train_dir)
    valid_data = train_transformer(valid_dir)
    test_data = train_transformer(test_dir)
    
    trainloader = data_loader(train_data)
    validloader = data_loader(valid_data, train=False)
    testloader = data_loader(test_data, train=False)
    
    model = primaryloader_model(architecture=args.arch)
    
    model.classifier = initial_classifier(model, hidden_units=args.hidden_units)
    
    device = check_gpu(gpu_arg=args.gpu);
    model.to(device);
    
    
    if type(args.learning_rate) == type(None):
        learning_rate = 0.001
        print("Learning = 0.001")
    else: learning_rate = args.learning_rate
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    print_every = 30
    steps = 0
    
    trained_model = network_trainer(model, trainloader, validloader,device, criterion, optimizer, args.epochs, print_every, steps)
    
    print("\nTraining process completed")
    
    validate_model(trained_model, testloader, device)
   
    initial_checkpoint(trained_model, args.save_dir, train_data)
if __name__ == '__main__': main()

