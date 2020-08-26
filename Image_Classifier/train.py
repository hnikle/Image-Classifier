
import numpy as np
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
import argparse

def validation(model, testloader, criterion):
    test_loss = 0
    accuracy = 0
    model.to(device)
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        output = model.forward(images)
        test_loss += criterion(output, labels).item()
        
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return test_loss, accuracy

def check_gpu(gpu_arg):
    if not gpu_arg:
        return torch.device("cpu")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    
    if device == "cpu":
        print("CUDA was not found.")
    return device

def primary_model(architecture="vgg16"):
    model = models.vgg16(pretrained=True)
    model.name = "vgg16"
        
    for param in model.parameters():
        param.requires_grad = False 
    return model

parser = argparse.ArgumentParser()
parser.add_argument('--arch', type = str, default = 'vgg16')
parser.add_argument('--save_dir', dest= 'save_dir', type = str, default = './checkpoint.pth')
parser.add_argument('--learning_rate', type = float, default = 0.001)
parser.add_argument('--hidden_units', type = int, action= 'store', dest = 'hidden_units', default = 512)
parser.add_argument('--epochs', type = int, default = 5)
parser.add_argument('--gpu', dest="gpu", action="store", default="gpu")

args = parser.parse_args()


data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomResizedCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32)

model = primary_model(architecture=args.arch)
hidden_units = args.hidden_units


classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(25088, hidden_units)),
                                        ('relu', nn.ReLU()),
                                        ('dropout', nn.Dropout(0.5)),
                                        ('fc2', nn.Linear(hidden_units,102)),
                                        ('output', nn.LogSoftmax(dim=1))]))
model.classifier = classifier
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), args.learning_rate)

epochs = args.epochs
device = check_gpu(gpu_arg=args.gpu)
model.to(device)

steps = 0
print_every = 30

for epoch in range(epochs):
    model.train()
    running_loss = 0
    
    for images, labels in iter(trainloader):
        steps += 1

        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        
        output = model.forward(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if steps % print_every == 0:
            model.eval()
            
            with torch.no_grad():
                test_loss, accuracy = validation(model, testloader, criterion)
            
            print("Epoch: {}/{} ... ".format(epoch+1, epochs),
                  "Training Loss: {:.3f} ... ".format(running_loss/print_every),
                  "Test Loss: {:.3f} ... ".format(test_loss/len(testloader)),
                  "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))
            
            running_loss = 0
            model.train()

model.class_to_idx = train_data.class_to_idx

checkpoint = {'input_size': 25088,
              'output_size': 102,
              'epochs': epochs,
              'model': getattr(models, args.arch)(pretrained=True),
              'classifier': classifier,
              'optimizer': optimizer.state_dict(),
              'class_to_idx': model.class_to_idx,
              'state_dict': model.state_dict()}

torch.save(checkpoint, args.save_dir)