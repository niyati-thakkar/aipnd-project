import argparse

import torch
from torchvision import datasets, transforms, models
from PIL import Image
from torch import nn
from torch import optim

# parser
parser = argparse.ArgumentParser(description='Predicting flower name from an image along with the probability of that name.')
parser.add_argument('image_path', help='Path to image')
parser.add_argument('checkpoint', help='Given checkpoint of a network')
parser.add_argument('--top_k', help='Return top k most likely classes')
parser.add_argument('--category_names', help='Use a mapping of categories to real names')
parser.add_argument('--gpu', help='Use GPU for inference', action='store_true')

args = parser.parse_args()

# user variables
top_k = 1 if args.top_k is None else int(args.top_k)
category_names = "cat_to_name.json" if args.category_names is None else args.category_names
gpu = False if args.gpu is None else True

# function for loading the model
def load_model(filepath):
    print("Loading and building model from {}".format(filepath))

    checkpoint = torch.load(filepath)
    model = build_network(checkpoint['architecture'], checkpoint['hidden_units'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx'] 
    
    return model
    
    
    # predicting the output with image path and topk variable
def predict(processed_image, model, topk):
    model.eval()
    with torch.no_grad():
        logps = model.forward(processed_image.unsqueeze(0))
        ps = torch.exp(logps)
        probs, labels = ps.topk(topk, dim=1)
        
        class_to_idx_inv = {model.class_to_idx[i]: i for i in model.class_to_idx}
        classes = list()
    
        for label in labels.numpy()[0]:
            classes.append(class_to_idx_inv[label])
        
        return probs.numpy()[0], classes
    
    # driver code
def main():
    
    # getting model
    model = load_model(args.checkpoint)
    print(model)

    
    # predicting output
    probs, predict_classes = predict(data_management.process_image(args.image_path), model, top_k)

    with open(category_names, 'r') as f:
        cat_to_name = json.load(f, strict=False)

    classes = []

    # getting class value
    for predict_class in predict_classes:
        classes.append(cat_to_name[predict_class])

    print(probs)
    print(classes)
if __name__ == '__main__': main()