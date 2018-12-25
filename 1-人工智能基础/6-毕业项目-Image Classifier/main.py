



import os
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models

import numpy as np
import pandas as pd

from PIL import Image
import tensorflow as tf

from collections import OrderedDict
import json

#测试网络               
def check_accuracy_on_test(model,dataloaders,device):   
    # TODO: Do validation on the test set
    correct = 0
    total = 0

    model.to(device)
    with torch.no_grad():
        for data in dataloaders['test']:
            images, labels = data
            images, labels = images.to(device),labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data,1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the {} test images: {:.1f}'.format(total,(100 * correct / total)))

#保存检查点
def save_checkpoint(model,save_dir,image_datasets):
    #torch.save(model, save_dir)
    
    checkpoint = {'state_dict':model.state_dict(),
         'image_datasets':image_datasets['train'].class_to_idx,
         'classifier':model.classifier      
        }
    torch.save(checkpoint, save_dir)
    
#加载检查点
def load_checkpoint(filepath):

    
    model = models.vgg16(pretrained=True)
    
    checkpoint = torch.load(filepath)
    model.idx_to_class = checkpoint['image_datasets']
    model.classifier = checkpoint['classifier']
    
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    im = Image.open(image)

    transform = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    '''
    transforms.Normalize使用如下公式进行归一化：

    channel=（channel-mean）/std
    '''
    return transform(im)

def predict(image_path, model, gpu,topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    
    if gpu and torch.cuda.is_available():
        model.to('cuda')
    else:
        model.to('cpu')
    
    inputs = process_image(image_path).to('cuda')
    inputs.unsqueeze_(0)
        
    outputs = model(inputs)
    outputs = F.softmax(outputs,dim=1)
    
    probs, classes =outputs.topk(topk)
    return probs.tolist()[0],classes.tolist()[0]

def do_train(data_dir, save_dir,arch, learning_rate, hidden_units, epochs, gpu):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in ['train', 'test', 'valid']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64,
                                                shuffle=True)
                for x in ['train', 'test', 'valid']}


    # TODO: Build and train your network
    if arch == 'vgg13':
        model = models.vgg13(pretrained=True)
    elif arch == 'vgg16':
        model = models.vgg16(pretrained=True)


    
    classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(25088, hidden_units)),
                            ('relu', nn.ReLU()),
                            
                            ('fc2', nn.Linear(hidden_units, hidden_units//10)),
                            ('relu', nn.ReLU()),
                            
                            ('fc3', nn.Linear(hidden_units//10, 102)),
                            ('relu', nn.ReLU()),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))


    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = classifier
    # TODO: Train a model with a pre-trained network
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    
    if gpu:
        device = 'cuda'
    else:
        device = 'cpu'
    
    steps = 0

    # change to cuda
    model.to(device)
    print_every = 20

    for e in range(epochs):
        running_loss = 0

        for _, (inputs, labels) in enumerate(dataloaders['train']):
            steps += 1

            inputs, labels = inputs.to(device),labels.to(device)

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
            
                accuracy = 0
                test_loss = 0
                for _, (inputs, labels) in enumerate(dataloaders['valid']):

                    inputs, labels = inputs.to(device),labels.to(device)

                    # Forward and backward passes
                    output = model.forward(inputs)
                    test_loss += criterion(output, labels).data[0]
                    
                    ps = torch.exp(output).data
                    # Class with highest probability is our predicted class, compare with true label
                    equality = (labels.data == ps.max(1)[1])
                    # Accuracy is number of correct predictions divided by all predictions, just take the mean
                    accuracy += equality.type_as(torch.FloatTensor()).mean()

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                  "Test Loss: {:.3f}.. ".format(test_loss/len(dataloaders['valid'])),
                  "Test Accuracy: {:.3f}".format(accuracy/len(dataloaders['valid'])))
                running_loss = 0
                model.train()

    test = input("是否需要测试 y / n :")

    if test == 'y' or test == 'Y':
        print("测试中。。。")
        check_accuracy_on_test(model,dataloaders,device)
        
    save = input("是否需要保存y / n :")
    if save == 'y' or save == 'Y':
        print("保存中。。。")
        save_checkpoint(model,save_dir,image_datasets)
        print("保存成功")


def do_predict(image_path,checkpoint_path,top_k,category_names,gpu):

    

    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)

    model = load_checkpoint(checkpoint_path)
    probs, classes = predict(image_path, model,gpu,top_k)

    Top_prop = probs[0]
    label = []
    for i in classes:
        label.append(cat_to_name[str(i)])
    
    print("Top {} classes is:\n".format(top_k))
    
    for i in range(top_k):
         print("{:.1f}% is {}".format(100*probs[i],label[i]))
