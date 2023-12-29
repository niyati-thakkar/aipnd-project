# imports
import argparse

import torch
from torchvision import datasets, transforms, models
from PIL import Image
from torch import nn
from torch import optim

#parser 
# getting user inputs for various variables
parser = argparse.ArgumentParser(description='Training a neural network on a given dataset')
parser.add_argument('data_directory', help='Path to dataset on which the neural network should be trained on')
parser.add_argument('--save_dir', help='Path to directory where the checkpoint should be saved')
parser.add_argument('--arch', help='Network architecture (default \'vgg16\')')
parser.add_argument('--learning_rate', help='Learning rate')
parser.add_argument('--hidden_units', help='Number of hidden units')
parser.add_argument('--epochs', help='Number of epochs')
parser.add_argument('--gpu', help='Use GPU for training', action='store_true')

args = parser.parse_args()

# user data and default data
save_dir = './' if args.save_dir is None else args.save_dir
network_architecture = 'vgg16' if args.arch is None else args.arch
learning_rate = 0.0001 if args.learning_rate is None else float(args.learning_rate)
hidden_units = 512 if args.hidden_units is None else float(args.hidden_units)
epochs = 5 if args.epochs is None else int(args.epochs)
gpu = False if args.gpu is None else True

# gpu
# getting device variable
def check_gpu(gpu_arg):
    if not gpu_arg:
        return torch.device("cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        print("Using CPU.")
    return device

# load data

def load_data(path):
    print("Loading and preprocessing data from")
    
    train_dir = path + '/train'
    valid_dir = path + '/valid'
    test_dir = path + '/test'
    
    # Define transforms for the various sets such as training, validation, and testing sets
    train_transform = transforms.Compose([transforms.RandomRotation(50),
                                                  transforms.RandomResizedCrop(224),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize([0.485, 0.456, 0.406],
                                                                       [0.229, 0.224, 0.225])])

    valid_transform = transforms.Compose([transforms.Resize(255),
                                                    transforms.CenterCrop(224),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                                         [0.229, 0.224, 0.225])])

    test_transform = transforms.Compose([transforms.Resize(255),
                                                 transforms.CenterCrop(224),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize([0.485, 0.456, 0.406],
                                                                      [0.229, 0.224, 0.225])])

    # Loading the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform = train_transform)
    valid_data = datasets.ImageFolder(valid_dir, transform = valid_transform)
    test_data = datasets.ImageFolder(test_dir, transform = test_transform)

    # Using the image datasets and the trainforms to get dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle = True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size = 64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size = 64)
    
    print("Finished loading and preprocessing data.")
    
    return train_data, trainloader, validloader, testloader

# training model code
def build_network(architecture, hidden_units):
    print("Building network..")
    
    if architecture =='vgg16':
        model = models.vgg16(pretrained = True)
        input_units = 25088
    elif architecture =='vgg13':
        model = models.vgg13(pretrained = True)
        input_units = 25088
    elif architecture =='alexnet':
        model = models.alexnet(pretrained = True)
        input_units = 9216
        
    for param in model.parameters():
        param.requires_grad = False
    
    classifier = nn.Sequential(
              nn.Linear(in_features = int(input_units), out_features = int(hidden_units)),
              nn.ReLU(),
              nn.Dropout(p=0.2),
              nn.Linear(hidden_units, 256),
              nn.ReLU(),
              nn.Dropout(p=0.2),
              nn.Linear(256, 102),
              nn.LogSoftmax(dim = 1)
            )

    model.classifier = classifier
    
    print("Finished building the network.")
    
    return model

# trainning the network with funciton
def train_network(model, epochs, learning_rate, trainloader, validloader, device):
    print("Training the network..")

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)

    model.to(device)
    
    
    # Training the network to get the train loss
    steps = 0
    print_every = 10
    train_loss = 0

    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                valid_accuracy = 0

                model.eval()

                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)

                        logps = model(inputs)
                        batch_loss = criterion(logps, labels)
                        valid_loss += batch_loss.item()

                        # Calculating validation accuracy for dataset
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        valid_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}, "
                      f"Train loss: {train_loss/print_every:.3f}, "
                      f"Valid loss: {valid_loss/len(validloader):.3f}, "
                      f"Valid accuracy: {valid_accuracy/len(validloader):.3f}")

                train_loss = 0

                model.train()

    print("Done training network.")            
    
    return model, criterion

def evaluate_model(model, testloader, criterion, device):
    print("Testing network...")
   
    
    # Validation on the test dataset
    test_loss = 0
    test_accuracy = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)

            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)
            test_loss += batch_loss.item()

            # Calculate accuracy on test dataset
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"Test loss: {test_loss/len(testloader):.3f}, "
          f"Test accuracy: {test_accuracy/len(testloader):.3f}")
    running_loss = 0
    
    print("Finished testing network.")
    
def save_model(model, architecture, hidden_units, epochs, learning_rate, save_dir):
    print("Saving model ... epochs: {}, learning_rate: {}, save_dir: {}".format(epochs, learning_rate, save_dir))
    checkpoint = {
        'architecture': architecture,
        'hidden_units': hidden_units,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'model_state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx
    }
    
    checkpoint_path = save_dir + "checkpoint.pth"

    torch.save(checkpoint, checkpoint_path)
    
    print("Model saved to {}".format(checkpoint_path))
    
def load_model(filepath):
    print("Loading and building model from {}".format(filepath))

    checkpoint = torch.load(filepath)
    model = build_network(checkpoint['architecture'], checkpoint['hidden_units'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx'] 
    
    return model

# executing all the functions
def main():
    # getting data and data loaders
    train_data, trainloader, validloader, testloader = load_data(args.data_directory) 
    device = check_gpu(gpu)

    # building network and training it
    model = build_network(network_architecture, hidden_units)
    model.class_to_idx = train_data.class_to_idx

    model, criterion = train_network(model, epochs, learning_rate, trainloader, validloader, device)
    
    # saving the model in checkpoint
    evaluate_model(model, testloader, criterion, device)
    save_model(model, network_architecture, hidden_units, epochs, learning_rate, save_dir)
if __name__ == '__main__': main()