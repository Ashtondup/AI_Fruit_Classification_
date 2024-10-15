import torch
from torchvision import datasets, transforms, utils
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve, auc, f1_score, accuracy_score
from sklearn.preprocessing import label_binarize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Paths for datasets
fruit_train = '/content/drive/My Drive/notebook_data_output/dataset3/train'
fruit_test = '/content/drive/My Drive/notebook_data_output/dataset3/test'
data_dir = '/content/drive/My Drive/notebook_data_output/dataset3'
print(torch.cuda.device_count())

# Data transforms
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

# Data loaders
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

# Model definition
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

net = Net().to(device)

# Load model if available
model_path = '/content/drive/My Drive/AshtonClolab/mobile/fruit_classifier.pth'
if os.path.exists(model_path):
    net.load_state_dict(torch.load(model_path))
    print("Loaded saved model.")

# Paths to save metrics
roc_auc_file_path = '/content/drive/My Drive/AshtonClolab/Metrics/ROC-AUC.txt'
f1_score_file_path = '/content/drive/My Drive/AshtonClolab/Metrics/F1_Score.txt'
accuracy_file_path = '/content/drive/My Drive/AshtonClolab/Metrics/Accuracy.txt'
loss_file_path = '/content/drive/My Drive/AshtonClolab/Metrics/Loss.txt'
os.makedirs(os.path.dirname(roc_auc_file_path), exist_ok=True)

# Training parameters
EPOCHS = 10
patience = 10
optimizer = optim.Adam(net.parameters(), lr=0.001)
cross_el = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.1)
best_loss = float('inf')
no_improvement_epochs = 0

# Training loop
for epoch in range(EPOCHS):
    print(f"Epoch: {epoch + 1}")
    net.train()
    running_loss = 0.0

    # Training phase
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

    # Save loss
    with open(loss_file_path, 'a') as loss_file:
        loss_file.write(f"Epoch {epoch + 1}: Loss = {average_loss}\n")

    # Evaluation phase
    net.eval()
    y_true = []
    y_pred = []
    y_prob = []

    with torch.no_grad():
        for data in data_loader['test']:
            x, y = data[0].to(device), data[1].to(device)
            output = net(x)
            probs = F.softmax(output, dim=1)
            _, preds = torch.max(output, 1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    # Compute accuracy
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy for Epoch {epoch + 1}: {acc}")
    with open(accuracy_file_path, 'a') as accuracy_file:
        accuracy_file.write(f"Epoch {epoch + 1}: Accuracy = {acc}\n")

    # Compute F1 Score (macro-average)
    f1 = f1_score(y_true, y_pred, average='macro')
    print(f"F1 Score for Epoch {epoch + 1}: {f1}")
    with open(f1_score_file_path, 'a') as f1_file:
        f1_file.write(f"Epoch {epoch + 1}: F1 Score = {f1}\n")

    # Binarize the labels for multi-class ROC calculation
    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))

    # Compute ROC curve, TPR, FPR for all classes combined (one-vs-rest approach)
    fpr, tpr, _ = roc_curve(y_true_bin.ravel(), np.array(y_prob).ravel())
    roc_auc = auc(fpr, tpr)

    # Save overall TPR, FPR, and AUC
    with open(roc_auc_file_path, 'a') as roc_auc_file:
        roc_auc_file.write(f"Epoch {epoch + 1}:\n")
        roc_auc_file.write(f"TPR: {tpr}, FPR: {fpr}, AUC: {roc_auc:.2f}\n\n")

    # Early stopping and learning rate adjustment
    if average_loss < best_loss:
        best_loss = average_loss
        no_improvement_epochs = 0
        torch.save(net.state_dict(), model_path)
        print(f"Best model saved after epoch {epoch + 1}")
    else:
        no_improvement_epochs += 1
        print(f"No improvement for {no_improvement_epochs} epochs")

    scheduler.step(average_loss)

    if no_improvement_epochs >= patience:
        print("Early stopping triggered")
        break

# Save final model
torch.save(net.state_dict(), model_path)
