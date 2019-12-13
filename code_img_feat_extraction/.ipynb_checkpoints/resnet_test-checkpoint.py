from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
plt.ion() # interactive mode
import argparse


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """
    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


def get_test_dataloader(data_dir, batch_size, img_dir):     
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_dataset = ImageFolderWithPaths(os.path.join(data_dir, img_dir), data_transform)
    test_dataloader = torch.utils.data.DataLoader(image_dataset, batch_size, shuffle=False, num_workers=4)
    
    class_names = image_dataset.classes
    print(class_names)  # ensure that the labels loaded are in the order as the saved model

    return test_dataloader, len(image_dataset)


def eval_model(test_dataloader, model, device, len_dataset):
    model.eval()
    running_corrects = 0.0
    with torch.no_grad():
        for i, (inputs, labels, img_paths) in enumerate(test_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            if i%1000 == 0:
                print("done eval till i", i)
            print(i, running_corrects)
    print('finished running.')
    epoch_acc = running_corrects.double() / len_dataset
    print("top-1 acc:", epoch_acc.item(), running_corrects.item(), len_dataset)
    
    
def main():
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('--batch_size', type=int, help='batch size', default=64)
    parser.add_argument('--data_dir', type=str, help='data dir', default='../datasets/food-101')
    parser.add_argument('--img_dir', type=str, help='img dir', default='test')
    parser.add_argument('--model_dir', type=str, help='model dir', default='../trained_models/test_food101n')
    parser.add_argument('--model_path', type=str, help='model path', default='model.pt_epoch_9')
    args = parser.parse_args()
    
    print(args)
    if not os.path.isdir(args.model_dir):
        os.makedirs(args.model_dir)
        print("made model directory.")
    save_model_path = args.model_dir + '/' + args.model_path
    model_ft = torch.load(save_model_path)
   
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print('..start program..')
    test_dataloader, len_dataset = get_test_dataloader(args.data_dir, args.batch_size, args.img_dir)
    model_ft = model_ft.to(device)
    
    model_ft = eval_model(test_dataloader, model_ft, device, len_dataset)
    # visualize_model(model_ft)

    print("..end program..")

if __name__=='__main__':
    main()
    
 