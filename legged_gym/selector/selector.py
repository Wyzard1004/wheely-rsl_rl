import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

import tensorboard
from torch.utils.tensorboard import SummaryWriter
import pickle
import os
import time

# Parameters
input_size = 121
hidden_sizes = [256, 128, 64, 32]
output_size = 4

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

if torch.cuda.is_available(): 
    dev = "cuda:0"
    print("using gpu")
else:
    dev = "cpu"
    print("using cpu")

device = torch.device(dev)



# Define the model
model = nn.Sequential(
    nn.Linear(input_size, hidden_sizes[0]),
    nn.BatchNorm1d(hidden_sizes[0]),
    nn.ReLU(),
    nn.Linear(hidden_sizes[0], hidden_sizes[1]),
    nn.BatchNorm1d(hidden_sizes[1]),
    nn.ReLU(),
    nn.Linear(hidden_sizes[1], hidden_sizes[2]),
    nn.BatchNorm1d(hidden_sizes[2]),
    nn.ReLU(),
    nn.Linear(hidden_sizes[2], hidden_sizes[3]),
    nn.BatchNorm1d(hidden_sizes[3]),
    nn.ReLU(),
    nn.Linear(hidden_sizes[3], output_size),
    nn.LogSoftmax(dim=1)
)
model=model.to(device)

class ArrayDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def preprocess_and_pickle_data(file_paths, data_dir):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    for i, (path, label) in enumerate(file_paths):
        data_list = []
        labels_list = []
        print(f"Processing and pickling {path}")
        with open(path, 'r', encoding='utf-8') as input_file:
            for line in input_file:
                numbers = [float(number) for number in line.strip().strip('[]').split()]
                if numbers:  # Only process if line is not empty
                    data_list.append(numbers)
                    labels_list.append(label)
        data_tensor = torch.tensor(data_list)
        data_tensor = data_tensor.to(device)
        labels_tensor = torch.tensor(labels_list, dtype=torch.long)
        labels_tensor = labels_tensor.to(device)
        output_file = os.path.join(data_dir, f"data_label_{i}.pkl")
        with open(output_file, 'wb') as f:
            pickle.dump((data_tensor, labels_tensor), f)

def load_and_process_data(data_dir):
    pickle_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.pkl')]
    if not pickle_files:  # If no pickle files found, preprocess and create them
        file_paths = [
            ("/home/william/legged_gym/legged_gym/terrain_data/cleaned_rough_slope.txt", 0),
            ("/home/william/legged_gym/legged_gym/terrain_data/cleaned_flat.txt", 1),
            ("/home/william/legged_gym/legged_gym/terrain_data/cleaned_stairs.txt", 2),
            ("/home/william/legged_gym/legged_gym/terrain_data/cleaned_slope.txt", 3)
        ]
        preprocess_and_pickle_data(file_paths, data_dir)
        print("pickling complete")
        pickle_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.pkl')]
    
    # Load all pickled data
    print("loading data")
    data_tensors = []
    label_tensors = []
    for file in pickle_files:
        with open(file, 'rb') as f:
            data, labels = pickle.load(f)
            data_tensors.append(data)
            label_tensors.append(labels)
    data = torch.cat(data_tensors, dim=0)
    labels = torch.cat(label_tensors, dim=0)
    print("data loaded")
    return data, labels

# Specify the output directory for your pickled data
data_dir = "/home/william/legged_gym/legged_gym/terrain_data/"
selector_dir = "/home/william/legged_gym/legged_gym/selector/"
data, labels = load_and_process_data(data_dir)
data = data.to(device)
labels = labels.to(device)

def evaluate_model_accuracy(model, dataloader, subset_size=1000):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    
    # Assuming subset_size is less than the total size of the dataset
    indices = torch.randperm(len(dataloader.dataset))[:subset_size]
    subset = torch.utils.data.Subset(dataloader.dataset, indices)
    subset_loader = DataLoader(subset, batch_size=dataloader.batch_size, shuffle=False)

    with torch.no_grad():
        for data, labels in subset_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

# Split the dataset into training and validation
def split_dataset(data, labels, train_fraction=0.75):
    print("splitting data")
    total_size = len(data)
    indices = torch.randperm(total_size)  # Shuffle indices
    indices = indices.to(device)
    train_size = int(total_size * train_fraction)
    
    # Use the shuffled indices to split the data and labels
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_data = data[train_indices]
    val_data = data[val_indices]
    
    train_labels = labels[train_indices]
    val_labels = labels[val_indices]
    print("finished splitting data")
    return train_data, train_labels, val_data, val_labels

# Split the data
train_data, train_labels, val_data, val_labels = split_dataset(data, labels)

# Create datasets and dataloaders
print("creating datasets")
train_dataset = ArrayDataset(train_data, train_labels)
val_dataset = ArrayDataset(val_data, val_labels)

print("creating dataloaders")
trainloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
valloader = DataLoader(val_dataset, batch_size=128, shuffle=True)

# Optimizer and loss function
print("creating optimizer and loss functions")
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
print("beginning training")
start_time = time.time()
best_accuracy = 0.0  # Track the best validation accuracy
best_model_path = os.path.join(selector_dir, 'best_model_state_dict.pth')

epochs = 1000
for e in range(epochs):
    epoch_start_time = time.time()
    model.train()
    running_loss = 0
    for data, labels in trainloader:
        optimizer.zero_grad()
        output = model(data)
        output = output.to(device)
        loss = criterion(output, labels)
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    writer.add_scalar("Loss/train", running_loss/len(trainloader), e)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, labels in valloader:
            output = model(data.float())
            loss = criterion(output, labels)
            val_loss += loss.item()
        writer.add_scalar("Loss/validation", val_loss/len(valloader), e)
        

    print(f"Epoch {e+1} - Training loss: {running_loss/len(trainloader)} - Validation loss: {val_loss/len(valloader)}")
    if (e + 1) % 5 == 0:  # Evaluate every 5 epochs
        val_accuracy = evaluate_model_accuracy(model, valloader, subset_size=1000)  # Adjust subset_size as needed
        print(f"Epoch {e+1} - Validation Accuracy: {val_accuracy}%")
        writer.add_scalar("Accuracy", val_accuracy/100, e)
        # Save the best model based on validation accuracy
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), best_model_path)
            print(f'New best model saved to {best_model_path} with validation accuracy: {best_accuracy}%')
    epoch_end_time = time.time()
    print(f"Epoch {e+1} took {epoch_end_time-epoch_start_time} seconds. Total Time Elapsed: {epoch_end_time-start_time} seconds")
    print()

# Save the final model's state dictionary at the end of training
final_model_path = os.path.join(selector_dir, 'final_model_state_dict.pth')
torch.save(model.state_dict(), final_model_path)
print(f'Final model saved to {final_model_path}')
writer.close()