import os
from PIL import Image
import torch
from torchvision import transforms,datasets
from torch.utils.data import Subset,DataLoader,Dataset
import numpy as np
import random
seed=12
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False
np.random.seed(seed)
random.seed(seed)
class ImageDataset(Dataset):
    def __init__(self,image_dir,transform=None):
        
        self.image_dir=image_dir
        self.image_filenames=[
            f for f in os.listdir(image_dir) if f.endswith(('.png','.jpg','.jpeg'))
        ]
        self.transform=transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self,idx):
        img_path=os.path.join(self.image_dir,self.image_filenames[idx])
        image=Image.open(img_path).convert("RGB")  
        if self.transform:
            image=self.transform(image)
        return image
    
class CustomDataset(Dataset):
    def __init__(self,dataset):
        self.dataset=dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,idx):
        image,label=self.dataset[idx]
        return image,int(label.item())  
def get_custom_dataset(new_dataset):
    dataset=CustomDataset(new_dataset)
    return dataset

def get_loaders(data_dir_L,batch_size=8,seed=12):
    
    transform=transforms.Compose([
        transforms.Resize((120,80)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
    ])

    
    dataset=datasets.ImageFolder(root=data_dir_L,transform=transform)
    labels=np.array([dataset.targets[i] for i in range(len(dataset))])

    
    train_indices,test_indices=[],[]
    for label in np.unique(labels):
        label_indices=np.where(labels == label)[0]
        np.random.shuffle(label_indices)
        split_idx=int(0.2 * len(label_indices))
        train_indices.extend(label_indices[:split_idx])
        test_indices.extend(label_indices[split_idx:])

    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)

    
    train_dataset=Subset(dataset,train_indices)
    test_dataset=Subset(dataset,test_indices)
    train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    test_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

    return train_loader,test_loader

def get_unlabel_loaders(data_dir_U,batch_size=8):
    transform=transforms.Compose([
        transforms.Resize((120,80)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
    ])
    dataset=ImageDataset(data_dir_U,transform)
    data_loader=DataLoader(dataset,batch_size=batch_size,shuffle=False)
    return data_loader