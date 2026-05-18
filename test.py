import torch
from model import CNN
from dataset import get_loaders

def test_model():
    
    data_dir_L='/mnt/c/Users/todds/Desktop/env/Final/train'

    
    _, test_loader=get_loaders(data_dir_L)

    
    model=CNN()
    model.load_state_dict(torch.load("model.pth"))
    model.eval()
    
    correct, total=0, 0
    correct_0, total_0=0, 0
    correct_1, total_1=0, 0
    with torch.no_grad():
        print(len(test_loader.dataset))
        for images, labels in test_loader:
            outputs=model(images)
            
            _, predicted=torch.max(outputs, 1)
            
            total+=labels.size(0)
            correct+=(predicted == labels).sum().item()
            total_0+=(labels == 0).sum().item()
            correct_0+=(((predicted == labels) & (labels == 0))).sum().item()
            total_1+=(labels == 1).sum().item()
            correct_1+=(((predicted == labels) & (labels == 1))).sum().item()

    
    accuracy=100 * correct / total
    c_0=100 * correct_0 / total_0 if total_0 > 0 else 0
    c_1=100 * correct_1 / total_1 if total_1 > 0 else 0
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Class 0 Accuracy: {c_0:.2f}%")
    print(f"Class 1 Accuracy: {c_1:.2f}%")
    print(total_0, correct_0, total_1, correct_1, total)

if __name__ == "__main__":
    test_model()