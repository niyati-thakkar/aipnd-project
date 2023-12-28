import argparse
import torch
import pandas as pd
import numpy as np
import seaborn as sns

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

from math import ceil
from train import check_gpu
from torchvision import models

def arg_parser():
    # Define command line arguments and parse them
    parser = argparse.ArgumentParser(description="predict.py")
    parser.add_argument('--image', type=str, help='Point to image file for prediction.', required=True)
    parser.add_argument('--checkpoint', type=str, help='Point to checkpoint file as str.', required=True)
    parser.add_argument('--top_k', type=int, help='Choose top K matches as int.')
    parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
    parser.add_argument('--gpu', default="gpu", action="store", dest="gpu")
    args = parser.parse_args()
    return args


def load_checkpoint(checkpoint_path):
    # Load checkpoint and return the model
    checkpoint = torch.load(checkpoint_path)
    
    model = models.vgg16(pretrained=True)
    model.name = "vgg16"
    
    for param in model.parameters():
        param.requires_grad = False

    # Load from checkpoint
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model


def process_image(image):
    # Process image for prediction
    img = PIL.Image.open(image)
    original_width, original_height = img.size

    if original_width < original_height:
        size = [256, 256**600]
    else: 
        size = [256**600, 256]
        
    img.thumbnail(size)
    center = original_width/4, original_height/4
    left, top, right, bottom = center[0]-(244/2), center[1]-(244/2), center[0]+(244/2), center[1]+(244/2)
    img = img.crop((left, top, right, bottom))
    
    numpy_img = np.array(img) / 255 

    # Normalize each color channel
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    numpy_img = (numpy_img - mean) / std
        
    # Set the color to the first channel
    numpy_img = numpy_img.transpose(2, 0, 1)
    
    return numpy_img


def predict(image_tensor, model, device, cat_to_name, topk=5):
    # Make prediction using the model
    model.eval()
    with torch.no_grad():
        image_tensor = torch.from_numpy(image_tensor).float()
        image_tensor = Variable(image_tensor.unsqueeze(0))
        image_tensor = image_tensor.to(device)
        
        output = model.forward(image_tensor)
        
        probabilities = torch.exp(output)
        top_probs, top_labels = probabilities.topk(topk)

    top_probs = top_probs.tolist()[0] 
    top_labels = top_labels.tolist()[0]

    # Convert indices to classes
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labels]
    top_flowers = [cat_to_name[lab] for lab in top_labels]

    return top_probs, top_labels, top_flowers


def print_probability(probs, flowers):
    # Print probabilities and corresponding flower names
    for i, j in enumerate(zip(flowers, probs)):
        print("Rank {}: Flower: {}, Likelihood: {}%".format(i+1, j[0], ceil(j[1]*100)))


def main():
    # Main function to run the prediction
    args = arg_parser()

    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    model = load_checkpoint(args.checkpoint)
    image_tensor = process_image(args.image)
    device = check_gpu(gpu_arg=args.gpu)

    top_probs, top_labels, top_flowers = predict(image_tensor, model, device, cat_to_name, args.top_k)

    print_probability(top_probs, top_flowers)


if __name__ == '__main__': main()