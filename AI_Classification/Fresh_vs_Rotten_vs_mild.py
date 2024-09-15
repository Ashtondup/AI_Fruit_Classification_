import torch
from torchvision import datasets, transforms, utils
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

fruit_train = 'C:/Users/User/Desktop/Code/AI_Project_Fruit_Classification_Jupyter/FruQ-multi/train'
fruit_test = 'C:/Users/User/Desktop/Code/AI_Project_Fruit_Classification_Jupyter/FruQ-multi/test'
data_dir = 'C:/Users/User/Desktop/Code/AI_Project_Fruit_Classification_Jupyter/FruQ-multi'
print(torch.cuda.device_count())

data_transform = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# Load datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transform[x]) for x in ['train', 'test']}

class_names = image_datasets['train'].classes

data_loader = {x: torch.utils.data.DataLoader(image_datasets[x], shuffle=True, batch_size=32, num_workers=0) for x in ['train', 'test']}

def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

inputs, classes = next(iter(data_loader['train']))
out = utils.make_grid(inputs)
imshow(out, title=[class_names[x] for x in classes])

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, 32)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dropout(F.max_pool2d(self.relu(self.bn1(self.conv1(x))), 2))
        x = self.dropout(F.max_pool2d(self.relu(self.bn2(self.conv2(x))), 2))
        x = self.dropout(F.max_pool2d(self.relu(self.bn3(self.conv3(x))), 2))
        x = self.dropout(F.max_pool2d(self.relu(self.bn4(self.conv4(x))), 2))
        x = torch.flatten(x, 1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

net = Net().to(device)  # Move the model to GPU

model_path = 'C:/Users/User/Desktop/Code/AI_Project_Fruit_Classification_Jupyter/Mobile apps/fruit_classifier.pth'
if os.path.exists(model_path):
    net.load_state_dict(torch.load(model_path))
    print("Loaded saved model.")

# Define the number of EPOCHS
EPOCHS = 1
patience = 10  # Early stopping patience

# Initialize optimizer and learning rate scheduler
optimizer = optim.Adam(net.parameters(), lr=0.001)
cross_el = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.1)

best_loss = float('inf')
no_improvement_epochs = 0

# Training loop
for epoch in range(EPOCHS):
    print(f"Epoch: {epoch + 1}")
    net.train()  # Set the model to training mode
    running_loss = 0.0  # Initialize running loss for the epoch

    for batch_idx, data in enumerate(data_loader['train']):
        x, y = data[0].to(device), data[1].to(device)  # Move data to GPU
        optimizer.zero_grad()  # Clear gradients from the previous step
        output = net(x)  # Forward pass
        loss = cross_el(output, y)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update model parameters

        running_loss += loss.item()  # Accumulate loss
        if (batch_idx + 1) % 100 == 0:
            print(f"Batch {batch_idx + 1}/{len(data_loader['train'])}, Loss: {loss.item()}")

    average_loss = running_loss / len(data_loader['train'])
    print(f"Average Loss for Epoch {epoch + 1}: {average_loss}")

    # Check for early stopping
    if average_loss < best_loss:
        best_loss = average_loss
        no_improvement_epochs = 0
        # Save the best model
        torch.save(net.state_dict(), model_path)
        print(f"Best model saved after epoch {epoch + 1}")
    else:
        no_improvement_epochs += 1
        print(f"No improvement for {no_improvement_epochs} epochs")

    # Adjust learning rate based on validation loss
    scheduler.step(average_loss)

    if no_improvement_epochs >= patience:
        print("Early stopping triggered")
        break

# ROC-AUC Calculation (no plotting)
net.eval()
y_true = []
y_pred = []
y_prob = []  # To store predicted probabilities

with torch.no_grad():
    for data in data_loader['test']:
        x, y = data[0].to(device), data[1].to(device)  # Move data to GPU
        output = net(x)
        probs = F.softmax(output, dim=1)  # Apply softmax to get probabilities
        _, preds = torch.max(output, 1)

        y_true.extend(y.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        y_prob.extend(probs.cpu().numpy())  # Collect probabilities

# Binarize the labels for multi-class ROC calculation
y_true_bin = label_binarize(y_true, classes=range(len(class_names)))

# Compute ROC curve and ROC-AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(len(class_names)):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], np.array(y_prob)[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    print(f'Class {class_names[i]} AUC: {roc_auc[i]:.2f}')

# Calculate the overall AUC score (one-vs-rest)
macro_roc_auc_ovr = roc_auc_score(y_true_bin, y_prob, multi_class="ovr", average="macro")
print(f'Macro-Averaged ROC AUC (One-vs-Rest): {macro_roc_auc_ovr:.2f}')

# Accuracy and classification report
correct = sum([1 for i in range(len(y_true)) if y_true[i] == y_pred[i]])
total = len(y_true)
print(f'Accuracy: {round(correct/total, 3)}')

# Classification report and F1 score
print(classification_report(y_true, y_pred, zero_division=0))
f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
print(f"Weighted F1 Score: {f1}")

# Save the final model
torch.save(net.state_dict(), model_path)
