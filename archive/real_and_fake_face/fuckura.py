import os
import cv2
import numpy as np
base_path="archive/real_and_fake_face"
X=[]
y=[]
train_real_dir=os.path.join(base_path,"training_real")
train_fake_dir=os.path.join(base_path,"training_fake")

IMG_SIZE=(128,128)

for filename in os.listdir(train_real_dir):
    img_path=os.path.join(train_real_dir,filename)
    img = cv2.imread(img_path) 
    if img is not None:
        img = cv2.resize(img,IMG_SIZE)
        X.append(img)
        y.append(1)
for filename in os.listdir(train_fake_dir):
    img_path=os.path.join(train_fake_dir,filename)
    img = cv2.imread(img_path) 
    if img is not None:
        img = cv2.resize(img,IMG_SIZE)
        X.append(img)
        y.append(0)

X=np.array(X)
y=np.array(y)
X=X/255.0

x_test=[]
y_test=[]

base_path="archive/real_and_fake_face_detection"
test_real_dir=os.path.join(base_path,"training_real")
test_fake_dir=os.path.join(base_path,"training_fake")

base_path="/home/waasi/waasi/python/kura_fpga/archive/real_and_fake_face_detection/real_and_fake_face"
test_real_dir=os.path.join(base_path,"training_real")
test_fake_dir=os.path.join(base_path,"training_fake")
IMG_SIZE=(128,128)
for filename in os.listdir(test_real_dir):
    img_path=os.path.join(test_real_dir,filename)
    img = cv2.imread(img_path) 
    if img is not None:
        img = cv2.resize(img,IMG_SIZE)
        x_test.append(img)
        y_test.append(1)
for filename in os.listdir(test_fake_dir):
    img_path=os.path.join(test_fake_dir,filename)
    img = cv2.imread(img_path) 
    if img is not None:
        img = cv2.resize(img,IMG_SIZE)
        x_test.append(img)
        y_test.append(0)

x_test=np.array(x_test)
y_test=np.array(y_test)
x_test=x_test/255.0



import torch 
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms


x_train_tensor = torch.tensor(X, dtype=torch.float32).permute(0, 3, 1, 2)  # Change to (N, C, H, W)
y_train_tensor = torch.tensor(y, dtype=torch.long)  
x_test_tensor = torch.tensor(x_test, dtype=torch.float32).permute(0, 3, 1, 2)  # Change to (N, C, H, W)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)



class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)  # Adjusted for 128x128 input size
        self.fc2 = nn.Linear(128, 2)  # Binary classification
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 32 * 32)  # Flatten the tensor
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()   
print(f"Accuracy: {100 * correct / total}%")    
