import torch
import pandas as pd
from model import CNN
from dataset import get_loaders,get_unlabel_loaders,get_custom_dataset
from torch.utils.data import DataLoader,ConcatDataset,TensorDataset,Subset
import numpy as np
import matplotlib.pyplot as plt
from aov import Classifyer
file_dir="./thumbnails/"
def label_samples(selected_indices,unlabel_loader):
    
    labeled_samples=[]
    for idx in selected_indices:
        image,_=unlabel_loader.dataset[idx]

        
        plt.imshow(image.permute(1,2,0))  
        plt.title(f"Image Index: {idx}")
        plt.axis('off')
        plt.show()

        
        label=int(input(f"Enter the label for image index {idx}: "))
        labeled_samples.append((image,label))

    return labeled_samples



def update_loaders(labeled_loader,unlabel_loader,selected_indices,labeled_samples):
    
    
    concat_dataset=labeled_loader.dataset

    
    current_indices=list(range(len(concat_dataset)))
    current_indices=(
        list(labeled_loader.dataset.indices)
    )
    #print(len(current_indices))
    
    
    selected_indices=list(selected_indices)
    #print(len(selected_indices))
    
    new_indices=current_indices+selected_indices
    #print(len(new_indices))
    
    new_labeled_dataset=Subset(concat_dataset,new_indices)
    print(len(new_labeled_dataset))
    
    labeled_loader=DataLoader(new_labeled_dataset,batch_size=labeled_loader.batch_size,shuffle=True)
    print(len(labeled_loader))
    
    remaining_indices=list(set(range(len(unlabel_loader.dataset))) - set(selected_indices))
    new_unlabeled_dataset=Subset(unlabel_loader.dataset,remaining_indices)
    
    
    unlabel_loader=DataLoader(new_unlabeled_dataset,batch_size=unlabel_loader.batch_size,shuffle=False)
    print(len(unlabel_loader))
    return labeled_loader,unlabel_loader


def active_learning():
    data_dir_L = r'C:\Users\todds\Desktop\Final\train'
    data_dir_U = r'C:\Users\todds\Desktop\Final\thumbnails'
    
    #maybe change the path to where the data actualy is
    train_loader,_ =get_loaders(data_dir_L)
    unlabel_loader=get_unlabel_loaders(data_dir_U)
    
    model=CNN()
    criterion=torch.nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=0.0002)
    
    
    for round_idx in range(2):
        print(f"\nActive Learning Round {round_idx+1}")
        print("train",len(train_loader.dataset))
        
        model.train()
        for epoch in range(10): 
            running_loss=0.0
            for images,labels in train_loader:
                
                optimizer.zero_grad()
                outputs=model(images)
                loss=criterion(outputs,labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch [{epoch+1}],Loss: {running_loss / len(train_loader):.4f}")

        
        model.eval()
        uncertainties=[]
        indices=[]

        with torch.no_grad():
            for idx,image in enumerate(unlabel_loader.dataset):
                image=image.unsqueeze(0)
                outputs=model(image)
                probs=torch.softmax(outputs,dim=1)
                uncertainty=-torch.max(probs).item()
                uncertainties.append(uncertainty)
                indices.append(idx)

        selected_indices=np.argsort(uncertainties)[-10:]
        print(f"Selected Indices for Labeling: {selected_indices}")
        a=Classifyer("catalog.csv",selected_indices)
        df=pd.read_csv(file_dir+"catalog.csv")
        
        
        new_images,new_labels=[],[]
        for i in selected_indices:
            
            image=unlabel_loader.dataset[i]
            new_images.append(image.squeeze(0))
            if df.iloc[i]['aov'] == 1:
                new_labels.append(1)
            else:
                new_labels.append(0)
        
        i=df[((df.aov == 1) | (df.aov == 0))].index
        df=df.drop(i)
        
        new_images=torch.stack(new_images)  
        new_labels=torch.tensor(new_labels)  

        current_dataset=train_loader.dataset
        print("cur",len(current_dataset))
        new_dataset=TensorDataset(new_images,new_labels)
        new_dataset=get_custom_dataset(new_dataset)
        print("new",len(new_dataset))
        combined_dataset=ConcatDataset([current_dataset,new_dataset])
        print("comb",len(combined_dataset))

        
        train_loader=DataLoader(combined_dataset,batch_size=train_loader.batch_size,shuffle=True)

        remaining_indices=list(set(range(len(unlabel_loader.dataset)))-set(selected_indices))

        
        filtered_unlabeled_dataset=Subset(unlabel_loader.dataset,remaining_indices)

        
        unlabel_loader=DataLoader(filtered_unlabeled_dataset,batch_size=unlabel_loader.batch_size,shuffle=False)
        #labeled_samples=label_samples(selected_indices,unlabel_loader)
        print("unlabel",len(unlabel_loader.dataset))
        #update_loaders(train_loader,unlabel_loader,selected_indices,labeled_samples)

    torch.save(model.state_dict(),"model.pth")
    print("\nActive learning completed.")
    


if __name__ == "__main__":
    active_learning()