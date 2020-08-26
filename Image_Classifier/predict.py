import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from collections import OrderedDict
import PIL
from PIL import Image
import argparse
import json

def check_gpu(gpu_arg):
    if not gpu_arg:
        return torch.device("cpu")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if device == "cpu":
        print("CUDA was not found.")
    return device

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer = checkpoint['optimizer']
    epochs = checkpoint['epochs']
           
    return model, checkpoint['class_to_idx']

def process_image(image):
    image = Image.open(image)
    preprocess = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])
    
    image = preprocess(image)
    return image


def predict(image_path, model, topk):
    img = process_image(image_path)
    model.to(device)
    img = img.to(device)
    img_classes_dict = {v: k for k, v in model.class_to_idx.items()}

    with torch.no_grad():
        img.unsqueeze_(0)
        output = model.forward(img)
        ps = torch.exp(output)
        probs, classes = ps.topk(topk)
        probs, classes = probs[0].tolist(), classes[0].tolist()
        
        return_classes = []
        for cls in classes:
            return_classes.append(img_classes_dict[cls])
            
        return probs, return_classes

parser = argparse.ArgumentParser()
parser.add_argument('--image_path', type=str)
parser.add_argument('--checkpoint',type=str, required=True)
parser.add_argument('--topk', type=int, default=3)
parser.add_argument('--gpu', dest="gpu", action="store", default="gpu")    
args = parser.parse_args()

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

device = check_gpu(gpu_arg=args.gpu)
model, class_to_idx = load_checkpoint(args.checkpoint)

probs, classes = predict(args.image_path, model, args.topk)
labels = [cat_to_name[cls] for cls in classes]

print ('Classes: ', labels)
print('Probability: ', probs)

