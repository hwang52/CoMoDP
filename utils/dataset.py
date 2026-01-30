import torch
import torch.utils
import torchvision
from torchvision import datasets
from torchvision.transforms import transforms
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Subset
from torch.utils.data import ConcatDataset
from config import *
from utils.fgvc_aircraft import Aircraft 
from utils.fgvc_cub import Cub2011
from utils.fgvc_flower import FlowersDataset
from utils.fgvc_car import Cars
from utils.fgvc_dog import Dogs
from utils.fgvc_bird import NABirds
from torch.utils.data import random_split
from utils.vtab_dtd import DTDDataloader
from utils.mnistm import MNISTM
from utils.syn import SyntheticDigits
import numpy as np
from utils.cifarc import CIFAR10C, CIFAR100C


class GrayscaleToRGB(object):
    def __call__(self, img):
        if img.mode == 'L':
            img = img.convert("RGB")
        return img


class get_dataset(object):
    def __init__(self, args):
        super().__init__()
        self.args = args


    def get_caltech10_dataset(self):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.args.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        dataset = datasets.ImageFolder("./dataset/caltech10/", transform=transform)
        train_size, test_size = int(0.8*len(dataset)), len(dataset)-int(0.8*len(dataset))
        train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])
        testset_num = len(test_set)
        class_names = ['backpack', 'bike', 'calculator', 'headphone', 'keyboard', 'laptop', 'monitor', 'mouse', 'mug', 'projector']
        train_loader = DataLoader(train_set, batch_size=64, pin_memory=True, num_workers=12, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=64, pin_memory=True, num_workers=12, shuffle=False)
        return train_loader, test_loader, testset_num, class_names


    def get_amazon_dataset(self):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.args.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        dataset = datasets.ImageFolder("./dataset/amazon/", transform=transform)
        train_size, test_size = int(0.8*len(dataset)), len(dataset)-int(0.8*len(dataset))
        train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])
        testset_num = len(test_set)
        class_names = ['backpack', 'bike', 'calculator', 'headphone', 'keyboard', 'laptop', 'monitor', 'mouse', 'mug', 'projector']
        train_loader = DataLoader(train_set, batch_size=64, pin_memory=True, num_workers=12, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=64, pin_memory=True, num_workers=12, shuffle=False)
        return train_loader, test_loader, testset_num, class_names


    def get_webcam_dataset(self):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.args.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        dataset = datasets.ImageFolder("./dataset/webcam/", transform=transform)
        train_size, test_size = int(0.8*len(dataset)), len(dataset)-int(0.8*len(dataset))
        train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])
        testset_num = len(test_set)
        class_names = ['backpack', 'bike', 'calculator', 'headphone', 'keyboard', 'laptop', 'monitor', 'mouse', 'mug', 'projector']
        train_loader = DataLoader(train_set, batch_size=64, pin_memory=True, num_workers=12, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=64, pin_memory=True, num_workers=12, shuffle=False)
        return train_loader, test_loader, testset_num, class_names


    def get_dslr_dataset(self):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.args.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        dataset = datasets.ImageFolder("./dataset/dslr/", transform=transform)
        train_size, test_size = int(0.8*len(dataset)), len(dataset)-int(0.8*len(dataset))
        train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])
        testset_num = len(test_set)
        class_names = ['backpack', 'bike', 'calculator', 'headphone', 'keyboard', 'laptop', 'monitor', 'mouse', 'mug', 'projector']
        train_loader = DataLoader(train_set, batch_size=64, pin_memory=True, num_workers=12, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=64, pin_memory=True, num_workers=12, shuffle=False)
        return train_loader, test_loader, testset_num, class_names


    def get_cifar10_dataset(self): # Cifar10 10
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.args.image_size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.2435, 0.2616)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.args.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.2435, 0.2616)),
        ])
        train_set = datasets.CIFAR10('./dataset/cifar10/', train=True, download=True, transform=transform_train)
        test_set = datasets.CIFAR10('./dataset/cifar10/', train=False, transform=transform_test)
        testset_num = len(test_set)
        class_names = train_set.classes
        # random get 10% dataset
        subset_size = int(len(train_set) * 0.1)
        indices = list(range(len(train_set)))
        np.random.shuffle(indices)
        train_indices = indices[:subset_size]
        train_set = Subset(train_set, train_indices)

        if self.args.ddp:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True)
            train_loader = DataLoader(train_set, sampler=train_sampler, batch_size=self.args.batch_size_train, 
                                pin_memory=True, num_workers=16)
            test_sampler = torch.utils.data.DistributedSampler(test_set)
            test_loader = DataLoader(test_set, sampler=test_sampler, batch_size=self.args.batch_size_test, 
                                pin_memory=True, num_workers=16)
        else:
            train_loader = DataLoader(train_set, batch_size=self.args.batch_size_train, pin_memory=True, num_workers=12, shuffle=True)
            test_loader = DataLoader(test_set, batch_size=self.args.batch_size_test, pin_memory=True, num_workers=12, shuffle=False)
        return train_loader, test_loader, testset_num, class_names


    def get_cifar100_dataset(self): # Cifar100 100
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.args.image_size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.args.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        train_set = datasets.CIFAR100('./dataset/cifar100/', train=True, download=True, transform=transform_train)
        test_set = datasets.CIFAR100('./dataset/cifar100/', train=False, transform=transform_test)
        testset_num = len(test_set)
        class_names = train_set.classes
        # random get 10% dataset
        subset_size = int(len(train_set) * 0.1)
        indices = list(range(len(train_set)))
        np.random.shuffle(indices)
        train_indices = indices[:subset_size]
        train_set = Subset(train_set, train_indices)

        if self.args.ddp:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True)
            train_loader = DataLoader(train_set, sampler=train_sampler, batch_size=self.args.batch_size_train, 
                                pin_memory=True, num_workers=16)
            test_sampler = torch.utils.data.DistributedSampler(test_set)
            test_loader = DataLoader(test_set, sampler=test_sampler, batch_size=self.args.batch_size_test, 
                                pin_memory=True, num_workers=16)
        else:
            train_loader = DataLoader(train_set, batch_size=self.args.batch_size_train, shuffle=True)
            test_loader = DataLoader(test_set, batch_size=self.args.batch_size_test, shuffle=False)
        return train_loader, test_loader, testset_num, class_names
    

    def get_cifar10c_dataset(self): # Cifar10-C
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.args.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.2435, 0.2616)),
        ])
        cname = 'gaussian_noise'
        train_set = CIFAR10C('./dataset/cifar10c', cname, transform=transform)
        test_set = CIFAR10C('./dataset/cifar10c', cname, transform=transform)
        testset_num = len(test_set)
        org_set = datasets.CIFAR10('./dataset/cifar10/', train=True, download=True, transform=transform)
        class_names = org_set.classes

        if self.args.ddp:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True)
            train_loader = DataLoader(train_set, sampler=train_sampler, batch_size=self.args.batch_size_train, 
                                pin_memory=True, num_workers=16)
            test_sampler = torch.utils.data.DistributedSampler(test_set)
            test_loader = DataLoader(test_set, sampler=test_sampler, batch_size=self.args.batch_size_test, 
                                pin_memory=True, num_workers=16)
        else:
            train_loader = DataLoader(train_set, batch_size=self.args.batch_size_train, pin_memory=True, num_workers=12, shuffle=False)
            test_loader = DataLoader(test_set, batch_size=self.args.batch_size_test, pin_memory=True, num_workers=12, shuffle=False)
        return train_loader, test_loader, testset_num, class_names


    def get_cifar100c_dataset(self): # Cifar100-C
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.args.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.2435, 0.2616)),
        ])
        cname = 'gaussian_noise'
        train_set = CIFAR100C('./dataset/cifar100c', cname, transform=transform)
        test_set = CIFAR100C('./dataset/cifar100c', cname, transform=transform)
        testset_num = len(test_set)
        org_set = datasets.CIFAR100('./dataset/cifar100/', train=True, download=True, transform=transform)
        class_names = org_set.classes

        if self.args.ddp:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True)
            train_loader = DataLoader(train_set, sampler=train_sampler, batch_size=self.args.batch_size_train, 
                                pin_memory=True, num_workers=16)
            test_sampler = torch.utils.data.DistributedSampler(test_set)
            test_loader = DataLoader(test_set, sampler=test_sampler, batch_size=self.args.batch_size_test, 
                                pin_memory=True, num_workers=16)
        else:
            train_loader = DataLoader(train_set, batch_size=self.args.batch_size_train, pin_memory=True, num_workers=12, shuffle=False)
            test_loader = DataLoader(test_set, batch_size=self.args.batch_size_test, pin_memory=True, num_workers=12, shuffle=False)
        return train_loader, test_loader, testset_num, class_names


    def get_vtab_svhn_dataset(self): # SVHN 10 
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.args.image_size),
            transforms.RandomHorizontalFlip(0.5),
            # transforms.Pad(4),
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.args.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        ])
        train_set = datasets.SVHN('./dataset/svhn/', train=True, download=True, transform=transform_train)
        test_set = datasets.SVHN('./dataset/svhn/', train=False, transform=transform_test)
        testset_num = len(test_set)
        class_names = train_set.classes

        if self.args.ddp:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True)
            train_loader = DataLoader(train_set, sampler=train_sampler, batch_size=self.args.batch_size_train, 
                                pin_memory=True, num_workers=16)
            test_sampler = torch.utils.data.DistributedSampler(test_set)
            test_loader = DataLoader(test_set, sampler=test_sampler, batch_size=self.args.batch_size_test, 
                                pin_memory=True, num_workers=16)
        else:
            train_loader = DataLoader(train_set, batch_size=self.args.batch_size_train, 
                                pin_memory=True, num_workers=16, shuffle=True)
            test_loader = DataLoader(test_set, batch_size=self.args.batch_size_test, 
                                pin_memory=True, num_workers=16, shuffle=False)
        return train_loader, test_loader, testset_num, class_names


    def get_fgvc_aircraft_dataset(self): # FGVC-Aircraft 100 
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.args.image_size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.args.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        ])
        train_set = Aircraft('./dataset/aircraft', train=True, download=False, transform=transform_train)
        test_set = Aircraft('./dataset/aircraft', train=False, download=False, transform=transform_test)
        testset_num = len(test_set)
        class_names = train_set.find_classes()[2]
        for idx in range(len(class_names)):
            temp_name = class_names[idx].replace('\n', '')
            class_names[idx] = temp_name
        
        if self.args.ddp:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True)
            train_loader = DataLoader(train_set, sampler=train_sampler, batch_size=self.args.batch_size_train, 
                                pin_memory=True, num_workers=16)
            test_sampler = torch.utils.data.DistributedSampler(test_set)
            test_loader = DataLoader(test_set, sampler=test_sampler, batch_size=self.args.batch_size_test, 
                                pin_memory=True, num_workers=16)
        else:
            train_loader = DataLoader(train_set, batch_size=self.args.batch_size_train, 
                                pin_memory=True, num_workers=16, shuffle=True)
            test_loader = DataLoader(test_set, batch_size=self.args.batch_size_test, 
                                pin_memory=True, num_workers=16, shuffle=False)
        return train_loader, test_loader, testset_num, class_names


    def get_fgvc_cub_dataset(self): # Cub2011 200 
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.args.image_size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.args.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        train_set = Cub2011('./dataset/cub2011', train=True, download=True, transform=transform_train)
        test_set = Cub2011('./dataset/cub2011', train=False, download=True, transform=transform_test)
        testset_num = len(test_set)
        class_names = train_set.class_names
        for idx in range(len(class_names)):
            temp_name = class_names[idx].split('.')[1]
            class_names[idx] = temp_name
        
        if self.args.ddp:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True)
            train_loader = DataLoader(train_set, sampler=train_sampler, batch_size=self.args.batch_size_train, 
                                pin_memory=True, num_workers=16)
            test_sampler = torch.utils.data.DistributedSampler(test_set)
            test_loader = DataLoader(test_set, sampler=test_sampler, batch_size=self.args.batch_size_test, 
                                pin_memory=True, num_workers=16)
        else:
            train_loader = DataLoader(train_set, batch_size=self.args.batch_size_train, 
                                pin_memory=True, num_workers=16, shuffle=True)
            test_loader = DataLoader(test_set, batch_size=self.args.batch_size_test, 
                                pin_memory=True, num_workers=16, shuffle=False)
        return train_loader, test_loader, testset_num, class_names


    def get_fgvc_flower_dataset(self): # Flowers 102 
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.args.image_size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.args.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        train_dataset = FlowersDataset('./dataset/flower102/train', transform=transform_train)
        val_dataset = FlowersDataset('./dataset/flower102/valid', transform=transform_train)
        train_set = ConcatDataset([train_dataset, val_dataset])
        test_set = FlowersDataset('./dataset/flower102/test', transform=transform_test)
        testset_num = len(test_set)
        class_names = train_dataset.classes
        
        if self.args.ddp:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True)
            train_loader = DataLoader(train_set, sampler=train_sampler, batch_size=self.args.batch_size_train, 
                                pin_memory=True, num_workers=16)
            test_sampler = torch.utils.data.DistributedSampler(test_set)
            test_loader = DataLoader(test_set, sampler=test_sampler, batch_size=self.args.batch_size_test, 
                                pin_memory=True, num_workers=16)
        else:
            train_loader = DataLoader(train_set, batch_size=self.args.batch_size_train, 
                                pin_memory=True, num_workers=16, shuffle=True)
            test_loader = DataLoader(test_set, batch_size=self.args.batch_size_test, 
                                pin_memory=True, num_workers=16, shuffle=False)
        return train_loader, test_loader, testset_num, class_names
    

    def get_fgvc_car_dataset(self): # Cars 196 
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.args.image_size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.args.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        train_set = Cars('./dataset/cars', train=True, img_path='./dataset/cars/cars_train', transform=transform_train)
        test_set = Cars('./dataset/cars', train=False, img_path='./dataset/cars/cars_test', transform=transform_test)
        testset_num = len(test_set)
        class_names = train_set.class_names
        
        if self.args.ddp:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True)
            train_loader = DataLoader(train_set, sampler=train_sampler, batch_size=self.args.batch_size_train, 
                                pin_memory=True, num_workers=16)
            test_sampler = torch.utils.data.DistributedSampler(test_set)
            test_loader = DataLoader(test_set, sampler=test_sampler, batch_size=self.args.batch_size_test, 
                                pin_memory=True, num_workers=16)
        else:
            train_loader = DataLoader(train_set, batch_size=self.args.batch_size_train, 
                                pin_memory=True, num_workers=16, shuffle=True)
            test_loader = DataLoader(test_set, batch_size=self.args.batch_size_test, 
                                pin_memory=True, num_workers=16, shuffle=False)
        return train_loader, test_loader, testset_num, class_names
    

    def get_fgvc_dog_dataset(self): # Dog 120 
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.args.image_size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.args.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        train_set = Dogs('./dataset/dogs', train=True, download=False, transform=transform_train)
        test_set = Dogs('./dataset/dogs', train=False, download=False, transform=transform_test)
        testset_num = len(test_set)
        class_names = train_set.class_names
        
        if self.args.ddp:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True)
            train_loader = DataLoader(train_set, sampler=train_sampler, batch_size=self.args.batch_size_train, 
                                pin_memory=True, num_workers=16)
            test_sampler = torch.utils.data.DistributedSampler(test_set)
            test_loader = DataLoader(test_set, sampler=test_sampler, batch_size=self.args.batch_size_test, 
                                pin_memory=True, num_workers=16)
        else:
            train_loader = DataLoader(train_set, batch_size=self.args.batch_size_train, 
                                pin_memory=True, num_workers=16, shuffle=True)
            test_loader = DataLoader(test_set, batch_size=self.args.batch_size_test, 
                                pin_memory=True, num_workers=16, shuffle=False)
        return train_loader, test_loader, testset_num, class_names
    

    def get_fgvc_bird_dataset(self): # NABird 555 
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.args.image_size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.args.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        train_set = NABirds('./dataset/nabirds', train=True, transform=transform_train)
        test_set = NABirds('./dataset/nabirds', train=False, transform=transform_test)
        testset_num = len(test_set)
        class_names = train_set.label_names
        
        if self.args.ddp:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True)
            train_loader = DataLoader(train_set, sampler=train_sampler, batch_size=self.args.batch_size_train, 
                                pin_memory=True, num_workers=16)
            test_sampler = torch.utils.data.DistributedSampler(test_set)
            test_loader = DataLoader(test_set, sampler=test_sampler, batch_size=self.args.batch_size_test, 
                                pin_memory=True, num_workers=16)
        else:
            train_loader = DataLoader(train_set, batch_size=self.args.batch_size_train, 
                                pin_memory=True, num_workers=16, shuffle=True)
            test_loader = DataLoader(test_set, batch_size=self.args.batch_size_test, 
                                pin_memory=True, num_workers=16, shuffle=False)
        return train_loader, test_loader, testset_num, class_names 


    def get_vtab_caltech_dataset(self): # Caltech, 101
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.args.image_size),
            GrayscaleToRGB(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        data_set = datasets.Caltech101(root='./dataset', transform=transform_train, download=True)
        class_names = data_set.categories
        TRAIN_SIZE = 0.8
        train_size = int(TRAIN_SIZE * len(data_set))
        test_size = len(data_set) - train_size
        train_set, test_set = random_split(data_set, [train_size, test_size])
        testset_num = test_size
        if self.args.ddp:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True)
            train_loader = DataLoader(train_set, sampler=train_sampler, batch_size=self.args.batch_size_train, 
                                pin_memory=True, num_workers=16)
            test_sampler = torch.utils.data.DistributedSampler(test_set)
            test_loader = DataLoader(test_set, sampler=test_sampler, batch_size=self.args.batch_size_test, 
                                pin_memory=True, num_workers=16)
        else:
            train_loader = DataLoader(train_set, batch_size=self.args.batch_size_train, 
                                pin_memory=True, num_workers=16, shuffle=True)
            test_loader = DataLoader(test_set, batch_size=self.args.batch_size_test, 
                                pin_memory=True, num_workers=16, shuffle=False)
        return train_loader, test_loader, testset_num, class_names
    

    def get_vtab_dtd_dataset(self): # DTD, 47
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.args.image_size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            normalize,
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.args.image_size),
            transforms.ToTensor(),
            normalize,
        ])
        train_set = DTDDataloader(transform_train, train=True)
        test_set = DTDDataloader(transform_test, train=False)
        testset_num = len(test_set)
        class_names = train_set.classes
        if self.args.ddp:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True)
            train_loader = DataLoader(train_set, sampler=train_sampler, batch_size=self.args.batch_size_train, 
                                pin_memory=True, num_workers=8)
            test_sampler = torch.utils.data.DistributedSampler(test_set)
            test_loader = DataLoader(test_set, sampler=test_sampler, batch_size=self.args.batch_size_test, 
                                pin_memory=True, num_workers=8)
        else:
            train_loader = DataLoader(train_set, batch_size=self.args.batch_size_train, 
                                pin_memory=True, num_workers=8, shuffle=True)
            test_loader = DataLoader(test_set, batch_size=self.args.batch_size_test, 
                                pin_memory=True, num_workers=8, shuffle=False)
        return train_loader, test_loader, testset_num, class_names


    def get_vtab_flower_dataset(self): # Flower, 102
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.args.image_size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            normalize,
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.args.image_size),
            transforms.ToTensor(),
            normalize,
        ])
        train_set = datasets.Flowers102(root='./dataset/oxford_flower/', transform=transform_train, split='train', download=True)
        test_set = datasets.Flowers102(root='./dataset/oxford_flower/', transform=transform_test, split='test', download=True)
        testset_num = len(test_set)
        import json
        with open('./utils/flower_name.json', 'r') as f:
            flower_names = json.load(f)
        flower_sorted_data = {k: flower_names[k] for k in sorted(flower_names, key=int)}
        class_names = list(flower_sorted_data.values())

        if self.args.ddp:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True)
            train_loader = DataLoader(train_set, sampler=train_sampler, batch_size=self.args.batch_size_train, 
                                pin_memory=True, num_workers=8)
            test_sampler = torch.utils.data.DistributedSampler(test_set)
            test_loader = DataLoader(test_set, sampler=test_sampler, batch_size=self.args.batch_size_test, 
                                pin_memory=True, num_workers=8)
        else:
            train_loader = DataLoader(train_set, batch_size=self.args.batch_size_train, 
                                pin_memory=True, num_workers=8, shuffle=True)
            test_loader = DataLoader(test_set, batch_size=self.args.batch_size_test, 
                                pin_memory=True, num_workers=8, shuffle=False)
        return train_loader, test_loader, testset_num, class_names
    

    def get_vtab_pet_dataset(self): # Pet, 37
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.args.image_size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            normalize,
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.args.image_size),
            transforms.ToTensor(),
            normalize,
        ])
        train_set = datasets.OxfordIIITPet(root='./dataset/oxford_pet/', transform=transform_train, split='trainval', download=True)
        test_set = datasets.OxfordIIITPet(root='./dataset/oxford_pet/', transform=transform_test, split='test', download=True)
        testset_num = len(test_set)
        class_names = train_set.classes
        if self.args.ddp:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True)
            train_loader = DataLoader(train_set, sampler=train_sampler, batch_size=self.args.batch_size_train, 
                                pin_memory=True, num_workers=8)
            test_sampler = torch.utils.data.DistributedSampler(test_set)
            test_loader = DataLoader(test_set, sampler=test_sampler, batch_size=self.args.batch_size_test, 
                                pin_memory=True, num_workers=8)
        else:
            train_loader = DataLoader(train_set, batch_size=self.args.batch_size_train, 
                                pin_memory=True, num_workers=8, shuffle=True)
            test_loader = DataLoader(test_set, batch_size=self.args.batch_size_test, 
                                pin_memory=True, num_workers=8, shuffle=False)
        return train_loader, test_loader, testset_num, class_names
    

    def get_vtab_pcam_dataset(self): # PCAM, 2
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.args.image_size),
            # transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            normalize,
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.args.image_size),
            transforms.ToTensor(),
            normalize,
        ])
        train_set = datasets.PCAM(root='./dataset/pcam/', transform=transform_train, split='train', download=True)
        test_set = datasets.PCAM(root='./dataset/pcam/', transform=transform_test, split='test', download=True)
        testset_num = len(test_set)
        class_names = ['normal tissue', 'tumor tissue']
        if self.args.ddp:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True)
            train_loader = DataLoader(train_set, sampler=train_sampler, batch_size=self.args.batch_size_train, 
                                pin_memory=True, num_workers=8)
            test_sampler = torch.utils.data.DistributedSampler(test_set)
            test_loader = DataLoader(test_set, sampler=test_sampler, batch_size=self.args.batch_size_test, 
                                pin_memory=True, num_workers=8)
        else:
            train_loader = DataLoader(train_set, batch_size=self.args.batch_size_train, 
                                pin_memory=True, num_workers=8, shuffle=True)
            test_loader = DataLoader(test_set, batch_size=self.args.batch_size_test, 
                                pin_memory=True, num_workers=8, shuffle=False)
        return train_loader, test_loader, testset_num, class_names
    

    def get_vtab_sun_dataset(self): # Sun397, 397
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.args.image_size),
            transforms.ToTensor(),
            normalize,
        ])
        data_set = datasets.SUN397(root='./dataset/sun/', transform=transform_train, download=False)
        class_names = data_set.classes 
        TRAIN_SIZE = 0.8
        train_size = int(TRAIN_SIZE * len(data_set))
        test_size = len(data_set) - train_size
        train_set, test_set = random_split(data_set, [train_size, test_size])
        testset_num = test_size
        
        if self.args.ddp:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True)
            train_loader = DataLoader(train_set, sampler=train_sampler, batch_size=self.args.batch_size_train, 
                                pin_memory=True, num_workers=8)
            test_sampler = torch.utils.data.DistributedSampler(test_set)
            test_loader = DataLoader(test_set, sampler=test_sampler, batch_size=self.args.batch_size_test, 
                                pin_memory=True, num_workers=8)
        else:
            train_loader = DataLoader(train_set, batch_size=self.args.batch_size_train, 
                                pin_memory=True, num_workers=8, shuffle=True)
            test_loader = DataLoader(test_set, batch_size=self.args.batch_size_test, 
                                pin_memory=True, num_workers=8, shuffle=False)
        return train_loader, test_loader, testset_num, class_names


    def get_vtab_eurosat_dataset(self): # EuroSAT, 10
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.args.image_size),
            transforms.ToTensor(),
            normalize,
        ])
        data_set = datasets.EuroSAT(root='./dataset/eurosat/', transform=transform_train, download=True)
        class_names = data_set.classes 
        TRAIN_SIZE = 0.8
        train_size = int(TRAIN_SIZE * len(data_set))
        test_size = len(data_set) - train_size
        train_set, test_set = random_split(data_set, [train_size, test_size])
        testset_num = test_size
        
        if self.args.ddp:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True)
            train_loader = DataLoader(train_set, sampler=train_sampler, batch_size=self.args.batch_size_train, 
                                pin_memory=True, num_workers=8)
            test_sampler = torch.utils.data.DistributedSampler(test_set)
            test_loader = DataLoader(test_set, sampler=test_sampler, batch_size=self.args.batch_size_test, 
                                pin_memory=True, num_workers=8)
        else:
            train_loader = DataLoader(train_set, batch_size=self.args.batch_size_train, 
                                pin_memory=True, num_workers=8, shuffle=True)
            test_loader = DataLoader(test_set, batch_size=self.args.batch_size_test, 
                                pin_memory=True, num_workers=8, shuffle=False)
        return train_loader, test_loader, testset_num, class_names
    

    def get_digits_mnistm_dataset(self): # MNIST-M 10 
        torch.manual_seed(0)
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.args.image_size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.args.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        ])
        train_set = MNISTM('./dataset/mnistm/', train=True, download=True, transform=transform_train)
        test_set = MNISTM('./dataset/mnistm/', train=False, download=True, transform=transform_test)
        testset_num = len(test_set)
        class_names = ['picture with handwritten number 0','picture with handwritten number 1','picture with handwritten number 2',
                       'picture with handwritten number 3','picture with handwritten number 4','picture with handwritten number 5',
                       'picture with handwritten number 6','picture with handwritten number 7','picture with handwritten number 8',
                       'picture with handwritten number 9']
        # random get 10% dataset
        subset_size = int(len(train_set) * 0.1)
        indices = list(range(len(train_set)))
        np.random.shuffle(indices)
        train_indices = indices[:subset_size]
        train_set = Subset(train_set, train_indices)

        if self.args.ddp:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True)
            train_loader = DataLoader(train_set, sampler=train_sampler, batch_size=self.args.batch_size_train, 
                                pin_memory=True, num_workers=12)
            test_sampler = torch.utils.data.DistributedSampler(test_set)
            test_loader = DataLoader(test_set, sampler=test_sampler, batch_size=self.args.batch_size_test, 
                                pin_memory=True, num_workers=12)
        else:
            train_loader = DataLoader(train_set, batch_size=self.args.batch_size_train, 
                                pin_memory=True, num_workers=12, shuffle=True)
            test_loader = DataLoader(test_set, batch_size=self.args.batch_size_test, 
                                pin_memory=True, num_workers=12, shuffle=False)
        return train_loader, test_loader, testset_num, class_names
    

    def get_digits_svhn_dataset(self): # SVHN 10 
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.args.image_size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.args.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        ])
        train_set = datasets.SVHN('./dataset/svhn/', split='train', transform=transform_train)
        test_set = datasets.SVHN('./dataset/svhn/', split='test', transform=transform_test)
        testset_num = len(test_set)
        class_names = ['picture with street view house number 10','picture with street view house number 1','picture with street view house number 2',
                       'picture with street view house number 3','picture with street view house number 4','picture with street view house number 5',
                       'picture with street view house number 6','picture with street view house number 7','picture with street view house number 8',
                       'picture with street view house number 9']
        # random get 10% dataset
        subset_size = int(len(train_set) * 0.1)
        indices = list(range(len(train_set)))
        np.random.shuffle(indices)
        train_indices = indices[:subset_size]
        train_set = Subset(train_set, train_indices)

        if self.args.ddp:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True)
            train_loader = DataLoader(train_set, sampler=train_sampler, batch_size=self.args.batch_size_train, 
                                pin_memory=True, num_workers=12)
            test_sampler = torch.utils.data.DistributedSampler(test_set)
            test_loader = DataLoader(test_set, sampler=test_sampler, batch_size=self.args.batch_size_test, 
                                pin_memory=True, num_workers=12)
        else:
            train_loader = DataLoader(train_set, batch_size=self.args.batch_size_train, 
                                pin_memory=True, num_workers=12, shuffle=True)
            test_loader = DataLoader(test_set, batch_size=self.args.batch_size_test, 
                                pin_memory=True, num_workers=12, shuffle=False)
        return train_loader, test_loader, testset_num, class_names
    

    def get_digits_usps_dataset(self): # USPS 10 
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.args.image_size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.args.image_size),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        ])
        train_set = datasets.USPS('./dataset/usps/', train=True, download=True, transform=transform_train)
        test_set = datasets.USPS('./dataset/usps/', train=False, download=True, transform=transform_test)
        testset_num = len(test_set)
        class_names = ['picture with number 0','picture with number 1','picture with number 2',
                       'picture with number 3','picture with number 4','picture with number 5',
                       'picture with number 6','picture with number 7','picture with number 8',
                       'picture with number 9']
        # random get 10% dataset
        subset_size = int(len(train_set) * 0.1)
        indices = list(range(len(train_set)))
        np.random.shuffle(indices)
        train_indices = indices[:subset_size]
        train_set = Subset(train_set, train_indices)

        if self.args.ddp:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True)
            train_loader = DataLoader(train_set, sampler=train_sampler, batch_size=self.args.batch_size_train, 
                                pin_memory=True, num_workers=12)
            test_sampler = torch.utils.data.DistributedSampler(test_set)
            test_loader = DataLoader(test_set, sampler=test_sampler, batch_size=self.args.batch_size_test, 
                                pin_memory=True, num_workers=12)
        else:
            train_loader = DataLoader(train_set, batch_size=self.args.batch_size_train, 
                                pin_memory=True, num_workers=12, shuffle=True)
            test_loader = DataLoader(test_set, batch_size=self.args.batch_size_test, 
                                pin_memory=True, num_workers=12, shuffle=False)
        return train_loader, test_loader, testset_num, class_names
    

    def get_digits_syn_dataset(self): # SYN 10 
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.args.image_size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.args.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        ])
        train_set = SyntheticDigits('./dataset/syn/', train=True, download=True, transform=transform_train)
        test_set = SyntheticDigits('./dataset/syn/', train=False, download=True, transform=transform_test)
        testset_num = len(test_set)
        class_names = ['picture with number 0','picture with number 1','picture with number 2',
                       'picture with number 3','picture with number 4','picture with number 5',
                       'picture with number 6','picture with number 7','picture with number 8',
                       'picture with number 9']
        # random get 10% dataset
        subset_size = int(len(train_set) * 0.1)
        indices = list(range(len(train_set)))
        np.random.shuffle(indices)
        train_indices = indices[:subset_size]
        train_set = Subset(train_set, train_indices)


        if self.args.ddp:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True)
            train_loader = DataLoader(train_set, sampler=train_sampler, batch_size=self.args.batch_size_train, 
                                pin_memory=True, num_workers=12)
            test_sampler = torch.utils.data.DistributedSampler(test_set)
            test_loader = DataLoader(test_set, sampler=test_sampler, batch_size=self.args.batch_size_test, 
                                pin_memory=True, num_workers=12)
        else:
            train_loader = DataLoader(train_set, batch_size=self.args.batch_size_train, 
                                pin_memory=True, num_workers=12, shuffle=True)
            test_loader = DataLoader(test_set, batch_size=self.args.batch_size_test, 
                                pin_memory=True, num_workers=12, shuffle=False)
        return train_loader, test_loader, testset_num, class_names