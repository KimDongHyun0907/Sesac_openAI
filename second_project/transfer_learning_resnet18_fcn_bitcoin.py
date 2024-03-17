import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

def change_tensor(data):
    np_test_tensor = []
    for i in data:
        np_test = np.tile(i, (1,4))
        np_test = np.tile(np_test, (28,1))
        np_test = torch.tensor(np_test, dtype=torch.float32)
        np_test = np_test.expand(3,-1,-1)

        np_test_tensor.append(np_test)

    np_test_tensor = np.array(np_test_tensor)
    np_test_tensor = torch.tensor(np_test_tensor)
    return np_test_tensor

# 데이터셋 크기에 맞는 CNN 모델 불러오기 (예: ResNet18)
model1 = models.resnet18(pretrained=True)

# 모델의 마지막 레이어를 새로운 fully connected layer로 교체
num_features = model1.fc.in_features
model1.fc = nn.Linear(num_features, 2)  # 예시에서는 클래스가 2개라 가정

# GPU 사용이 가능하다면 GPU로 모델 이동
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model1 = model1.to(device)

# Loss function 및 optimizer 설정
criterion = nn.CrossEntropyLoss()
lr = 0.001
momentum = 0.95
optimizer = optim.SGD(model1.parameters(), lr=lr, momentum=momentum)

# 학습 데이터 및 레이블
xTrainName = "bitcoin/bitcoin_data/train/bitcoin/xTrain_stretch.pkl"
yTrainName = "bitcoin/bitcoin_data/train/bitcoin/yTrain_stretch.pkl"

with open(xTrainName,'rb') as f1:
    X = pickle.load(f1)

with open(yTrainName,'rb') as f2:
    y = pickle.load(f2)

# x_np = X.to_numpy()
# y_np = y.to_numpy()

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size= 0.1, shuffle=True, random_state=42)

x_train = change_tensor(x_train)
x_test = change_tensor(x_test)
y_train = torch.tensor(np.tile(y_train, (2,)), dtype=torch.long)
y_test = torch.tensor(np.tile(y_test, (2,)), dtype=torch.long)

y_train = y_train.float()
y_test = y_test.float()

y_train = torch.argmax(y_train, dim=1)
y_test = torch.argmax(y_test, dim=1)

class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data
        label = self.labels
            
        return data[idx], label[idx]
    
train_dataset = CustomDataset(x_train, y_train)
batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 학습

num_epochs = 30
best_accuracy = 0.0
best_model = None

for epoch in range(num_epochs):
    model1.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        inputs = inputs.to(device)  
        outputs = model1(inputs)
        loss = criterion(outputs, labels.to(device))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    correct = 0
    total = 0
    
    # 모델을 evaluation 모드로 설정. 정확도 계산
    model1.eval()

    # 모델이 예측한 결과와 실제 레이블을 비교하여 정확도 계산
    with torch.no_grad():
        for inputs, labels in zip(x_test, y_test):
            inputs = inputs.unsqueeze(0).to(device)
            outputs = model1(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += 1
            correct += (predicted == labels.to(device)).sum().item()

    accuracy = correct / total

    # 정확도가 개선되지 않으면 count를 증가시킴
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model1.state_dict()
        count = 0
        
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(x_train)}, Accuracy: {accuracy}")

print(f'best accuracy : {best_accuracy}, batch_size = {batch_size}, lr = {lr}, momentum = {momentum}')

# 가장 높은 정확도의 모델을 저장
torch.save(best_model, 'best_model_resnet_fcn_bitcoin.pth')