from datasets.FFPPDataset import FFPPDataset, CelebValidateDataset
import torch 
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import argparse
import numpy as np
import random 
from tqdm import tqdm 

from models.resnet18 import Resnet, BasicBlock
from utils import save_plots

# Set seed.
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
np.random.seed(seed)
random.seed(seed)

# Training function
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0

    for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        counter = counter + 1
        image, labels = data 
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        #Forward pass 
        outputs = model(image)
        #Calculate loss 
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()

        # Calculate accuracy
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()
        # Back propagation
        loss.backward()
        # Update the weights
        optimizer.step()
    
    # Loss and accuracy for the complete epoch
    epoch_loss = train_running_loss / counter 
    epoch_acc = 100. * (train_running_correct / len(train_loader.dataset))

    return epoch_loss, epoch_acc

def validate(model, testloader, criterion, device):
    model.eval()
    print("Evaluation")
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0

    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1

            image, labels = data
            image = image.to(device)
            labels = labels.to(device)

            # Forward pass 
            outputs = model(image)
            # Calculate the loss 
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            # Calculate the accuracy 
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct = (preds == labels).sum().item()

    # Loss and accuracy for the complete epoch
    epoch_loss = valid_running_loss / counter 
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))

    return epoch_loss, epoch_acc
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--deepfake_folders", nargs="*", type=str, default=[])
    parser.add_argument("--original_folders", nargs="*", type=str, default=[])
    parser.add_argument("--test_root", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--model_dir", type=str)

    args = parser.parse_args()

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]) 

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    class A:
        deepfake_folders = args.deepfake_folders
        original_folders = args.original_folders
    class B: 
        test_root = args.test_root
    train_dataset = FFPPDataset(A, transform=train_transform)
    val_dataset = CelebValidateDataset(B, transform=train_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize model
    model = Resnet(img_channels=3, num_classes=2, block=BasicBlock, num_layers=18).to(device)
    plot_name = "resnet18"

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Lists to keep track of losses and accuracies.
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    # Start the training.
    epochs = args.epoch
    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch+1} of {epochs}")
        train_epoch_loss, train_epoch_acc = train(
            model, 
            train_loader, 
            optimizer, 
            criterion,
            device
        )
        valid_epoch_loss, valid_epoch_acc = validate(
            model, 
            val_loader, 
            criterion,
            device
        )
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)
        print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
        print('-'*50)
        
    # Save the loss and accuracy plots.
    save_plots(
        train_acc, 
        valid_acc, 
        train_loss, 
        valid_loss, 
        output_dir=args.output_dir,
        name=plot_name
    )
    torch.save(model, args.model_dir)
    print('TRAINING COMPLETE')