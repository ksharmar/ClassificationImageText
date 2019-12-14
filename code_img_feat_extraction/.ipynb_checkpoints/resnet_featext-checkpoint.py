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
import torch.utils.data as data
from PIL import Image
import pandas as pd


def default_flist_reader(flist, classnames_list):
    """ flist format: impath label\nimpath label\n """
    if classnames_list and classnames_list!='':
        classes = np.array(pd.read_csv(classnames_list, header=None)[0])
        classes_dict = dict(zip(classes, np.arange(len(classes))))
        df = pd.read_csv(flist, header=None, sep=' ')
        df[1] = df[1].apply(lambda x: classes_dict[x])
        imlist = map(tuple, df.values)
    else:
        df = pd.read_csv(flist, header=None, sep=' ')
        imlist = map(tuple,df.values)
        classes = np.arange(0, max(df[1])+1)
    return imlist, classes


class ImageFilelist(data.Dataset):

    def __init__(self, root, flist, classnames_list, transform=None, ret_path=False):
        self.root   = root
        self.imlist, self.classes = default_flist_reader(flist, classnames_list)
        self.transform = transform
        self.ret_path = ret_path
        # self.target_transform = target_transform

    def __getitem__(self, index):
        impath, label = self.imlist[index]
        img = Image.open(os.path.join(self.root, impath)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.ret_path:
            return img, label, impath 
        return img, label

    def __len__(self):
        return len(self.imlist)
    

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
    
    
def get_dataloaders(args):
    
    data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
#         transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    }
    image_datasets ={}
    dataloaders = {}
    # train (finetune)
    if args.tr_img_list and args.tr_img_list!='':
        image_datasets['train'] = ImageFilelist(args.tr_data_dir, args.tr_img_list, args.classname_list,
                                                data_transforms['train'], ret_path=False)
    else:   
        image_datasets['train'] = datasets.ImageFolder(os.path.join(args.tr_data_dir, args.tr_img_dir), 
                                                   data_transforms['train'])
    dataloaders['train'] = torch.utils.data.DataLoader(image_datasets['train'], 
                                                       batch_size=args.batch_size, shuffle=True, num_workers=4)

    # val/test (test in our case = No validation in training)
    if args.va_img_list and args.va_img_list!='':
        image_datasets['val'] = ImageFilelist(args.va_data_dir, args.va_img_list, args.classname_list,
                                              data_transforms['val'], ret_path=False)
    else:
        image_datasets['val'] =  datasets.ImageFolder(os.path.join(args.va_data_dir, args.va_img_dir), 
                                                   data_transforms['val'])
    dataloaders['val'] = torch.utils.data.DataLoader(image_datasets['val'], 
                                                       batch_size=args.batch_size, shuffle=False, num_workers=4)

    # feat_extraction (train set not shuffled)
    if args.ftext_img_list and args.ftext_img_list!='':
        feat_extraction_image_dataset = ImageFilelist(args.ftext_data_dir, args.ftext_img_list, args.classname_list,
                                                data_transforms['train'], ret_path=True)
    else:
        feat_extraction_image_dataset = ImageFolderWithPaths(os.path.join(args.ftext_data_dir, args.ftext_img_dir), 
                                                         data_transforms['train'])
    feat_extraction_data_loader = torch.utils.data.DataLoader(feat_extraction_image_dataset, 
                                                              batch_size=args.batch_size, 
                                                              shuffle=False, num_workers=4)

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    return dataloaders, dataset_sizes, class_names, feat_extraction_data_loader

    
def train_model(dataloaders, dataset_sizes, model, criterion, optimizer, scheduler,
                num_epochs, device, save_model_path):
    since = time.time()
    use_early_stop_val = True
    phases = ['train', 'val']
    
    if use_early_stop_val:
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        
        for phase in phases:
            if phase == 'train': model.train()  # Set model to training mode
            else: model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                if i % 1000 == 0:
                    print(i, loss.item())
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                # break
            if phase == 'train' and scheduler:
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if use_early_stop_val and phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            
        torch.save(model, save_model_path+"_epoch_"+str(epoch))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    
    if use_early_stop_val:
        print('Best val Acc: {:4f}'.format(best_acc))
        # load best model weights
        model.load_state_dict(best_model_wts)
    
    return model

def val_only(val_dataloader, val_datasize, model, criterion, device):
    since = time.time()
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    # Iterate over data.
    for i, (inputs, labels) in enumerate(val_dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        if i % 1000 == 0:
            print(i, loss.item())
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    epoch_loss = running_loss / val_datasize
    epoch_acc = running_corrects.double() / val_datasize
    print('{} Loss: {:.4f} Acc: {:.4f}'.format('val', epoch_loss, epoch_acc))
    time_elapsed = time.time() - since
    print('Eval complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))


def get_vector(feat_extraction_data_loader, model, save_emb_path, save_img_path, device, batch_size):
    was_training = model.training
    model.eval()
    extracted_embeddings = None
    extracted_img_paths = []
    f = open(save_emb_path, 'w')
    layer = model._modules.get('avgpool')
    # emb = torch.zeros(inputs.shape[0], 2048)
    def copy_data(m, i, o):
        for line in np.matrix(o.data.reshape(-1, 2048).cpu().numpy()):
            np.savetxt(f, line, fmt='%.3f')
        # emb.copy_(o.data.reshape(-1, 2048))
    # if extracted_embeddings is None: extracted_embeddings = emb
    # else: extracted_embeddings = torch.cat([extracted_embeddings, emb], 0)
    # np.savetxt(save_emb_path, extracted_embeddings.cpu().numpy())
    h = layer.register_forward_hook(copy_data)
     
    with torch.no_grad():
        for i, (inputs, labels, img_paths) in enumerate(feat_extraction_data_loader):
            # inputs = inputs[:10]
            # labels = labels[:10]
            extracted_img_paths += img_paths
            inputs = inputs.to(device)
            outputs = model(inputs)
            if i%1000 == 0:
                print("done extract at i", i)
            # break
        print("saving:")
        h.remove()
        np.savetxt(save_img_path, extracted_img_paths, fmt='%s')
        model.train(mode=was_training)
    print("saved embs")
    
    
def get_logits(feat_extraction_data_loader, model, device, save_logits_path):
    model.eval() 
    f = open(save_logits_path, 'w')
    with torch.no_grad():
        for i, (inputs, labels, img_paths) in enumerate(feat_extraction_data_loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            if i % 1000 == 0:
                print("done extract at i", i)
            for line in np.matrix(outputs.cpu().numpy()):
                np.savetxt(f, line, fmt='%.3f')
    print('saved logits')

    
def main():
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('--batch_size', type=int, help='batch size', default=64)
    parser.add_argument('--num_epochs', type=int, help='number of epochs', default=10)
    
    # this classname list is provided if string type labels in image list is given
    parser.add_argument('--classname_list', type=str, help='class names list', default='')
    
    # train data : train (run cases)
    parser.add_argument('--tr_data_dir', type=str, help='train data dir', default='')
    parser.add_argument('--tr_img_dir', type=str, help='train image dir', default='')
    parser.add_argument('--tr_img_list', type=str, help='train image list', default='')
    
    # val data (clean classification test set) : train, val (run cases)
    parser.add_argument('--va_data_dir', type=str, help='va data dir', default='')
    parser.add_argument('--va_img_dir', type=str, help='va image dir', default='')
    parser.add_argument('--va_img_list', type=str, help='va image list', default='')
    
    # feat extraction data : feat_ext, logit_ext (run cases)
    parser.add_argument('--ftext_data_dir', type=str, help='feat ext data dir', default='')
    parser.add_argument('--ftext_img_dir', type=str, help='feat ext image dir', default='')
    parser.add_argument('--ftext_img_list', type=str, help='feat ext image list', default='')
    
    parser.add_argument('--model_dir', type=str, help='model dir', default='')
    
    # SGD Optimizer: optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    parser.add_argument('--opt', type=str, help='opt', default='sgd')
    parser.add_argument('--lr', type=float, help='OPT: learning rate', default=0.001) 
    parser.add_argument('--momentum', type=float, help='SGD: momentum', default=0.9)
    parser.add_argument('--weightdecay', type=float, help='ADAM: weight decay', default=0.9)
    # Decay LR by 0.1 every 7 ep: exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    parser.add_argument('--gamma', type=float, help='LR sch: gamma', default=0.1) 
    parser.add_argument('--stepsize', type=float, help='LR sch: step size', default=7)
    
    parser.add_argument('--resnet', type=int, help='resnet layers', default=50)
    parser.add_argument('--run_case', type=str, help='logits_ext|feat_ext|train|val', default='')
    parser.add_argument('--saved_model_name', type=str, help='saved model name', default='')
        
    args = parser.parse_args()
    print(args)
    if not os.path.isdir(args.model_dir):
        os.makedirs(args.model_dir)
        print("made model directory.")
    
    save_model_path = args.model_dir + '/model.pt'
    save_emb_path = args.model_dir + '/extracted_emb.txt'
    save_img_path = args.model_dir + '/extracted_imgpaths.txt'
    save_logits_path = args.model_dir + '/val_logits.txt'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print('..start program..{}'.format(args.run_case))
    
    dataloaders, dataset_sizes, class_names, feat_extraction_data_loader = get_dataloaders(args)
    if args.run_case == 'train':
        if args.resnet == 18:
            model_ft = models.resnet18(pretrained=True)
        else:
            model_ft = models.resnet50(pretrained=True)
        # Add back for fixing the pretrained model (except last classification layer when finetuning) ========
        # for param in model_ft.parameters():
        #     param.requires_grad = False
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, len(class_names))
        model_ft = model_ft.to(device)
        criterion = nn.CrossEntropyLoss()

        if args.opt == 'sgd':
            optimizer_ft = optim.SGD(model_ft.parameters(), lr=args.lr, momentum=args.momentum)
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=args.stepsize, gamma=args.gamma)
        else:
            optimizer_ft = optim.Adam(model_ft.parameters(), lr=args.lr, weight_decay=0.99)
            exp_lr_scheduler = None

        model_ft = train_model(dataloaders, dataset_sizes, model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                               args.num_epochs, device, save_model_path)
        torch.save(model_ft, save_model_path)
        get_vector(feat_extraction_data_loader, model_ft, save_emb_path, save_img_path, device, args.batch_size)
    elif args.run_case == 'val':
        model_ft = torch.load(args.model_dir + '/' + args.saved_model_name)
        model_ft = model_ft.to(device)
        criterion = nn.CrossEntropyLoss()
        val_only(dataloaders['val'], dataset_sizes['val'], model_ft, criterion, device)
    elif args.run_case == 'feat_ext':
        model_ft = torch.load(args.model_dir + '/' + args.saved_model_name)
        model_ft = model_ft.to(device)
        get_vector(feat_extraction_data_loader, model_ft, save_emb_path, save_img_path, device, args.batch_size)
    elif args.run_case == 'logits_ext':
        model_ft = torch.load(args.model_dir + '/' + args.saved_model_name)
        model_ft = model_ft.to(device)
        get_logits(feat_extraction_data_loader, model_ft, device, save_logits_path)     
    print("..end program..")
    

    
if __name__=='__main__':
    main()