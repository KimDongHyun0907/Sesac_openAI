from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# pickle 파일 읽기
import pickle
import gzip

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

with gzip.open('Xdata.pickle', 'rb') as f:
    X_data = pickle.load(f)

with gzip.open('ydata.pickle', 'rb') as f:
    y_data = pickle.load(f)

tensor_list = [torch.tensor(array) for array in X_data.values]
X_data = torch.stack([torch.unsqueeze(tensor, dim=0) for tensor in tensor_list], dim=0)
y_data = torch.tensor(y_data)
y_data = torch.unsqueeze(y_data, dim=1)

class YourDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, index):
        X = self.X_data[index]
        y = self.y_data[index]

        return X, y

# 데이터를 훈련 세트와 검증 세트로 나눔
X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

batch_size = 8
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=6)
        self.pool = nn.MaxPool2d(kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=9, padding=5)
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=12)
        self.fc1 = nn.Linear(3, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1)
        self.dropout = nn.Dropout(p=0.5)  # Dropout 추가

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = torch.flatten(x, 1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  # Dropout
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)  # Dropout
        x = torch.sigmoid(self.fc3(x))  # 이진 분류를 위한 sigmoid 함수 사용
        return x

# 모델 인스턴스 생성
model = CNN().to(device)

# Loss function 및 Optimizer 정의
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Learning rate scheduler 정의
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

dataset = YourDataset(X_train, y_train)  # YourDataset은 실제 데이터셋을 로드하고 전처리하는 클래스입니다.
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  # batch_size는 한 번에 모델에 전달되는 데이터의 수입니다.

# 데이터셋 객체를 생성
val_dataset = YourDataset(X_val, y_val)  # X_val은 검증 데이터의 입력, y_val은 검증 데이터의 타겟
# DataLoader로 데이터셋을 배치 단위로 로드
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # batch_size는 한 번에 모델에 전달되는 데이터의 수입니다.

# 다음과 같이 모델을 저장할 경로를 지정합니다.
model_save_path = 'best_model.pth'
best_accuracy = 0.0

# 모델 학습
num_epochs = 100
for epoch in range(num_epochs):
    model.train()  # 모델을 학습 모드로 설정
    running_loss = 0.0
    tqdm_train_loader = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False)  # tqdm으로 train_loader 감싸기

    for inputs, labels in tqdm_train_loader:  # DataLoader에서 배치 단위로 데이터 로드
        optimizer.zero_grad()  # gradient 초기화
        inputs = inputs.float().to(device)
        labels = labels.float().to(device)
        outputs = model(inputs).float()  # 모델 예측
        loss = criterion(outputs.to(torch.float32), labels.to(torch.float32))  # 손실 계산
        loss.backward()  # 역전파
        optimizer.step()  # 옵티마이저 업데이트
        running_loss += loss.item() * inputs.size(0)
        tqdm_train_loader.set_postfix({'loss': running_loss / ((tqdm_train_loader.n + 1) * inputs.size(0))})  # tqdm 업데이트

    scheduler.step()  # Learning rate 스케줄링

    # 현재 epoch의 평균 손실 출력
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # 검증 데이터를 사용하여 모델 검증
    model.eval()  # 모델을 평가 모드로 설정
    correct = 0
    total = 0
    with torch.no_grad():  # 평가 과정에서는 gradient를 계산하지 않음
        for inputs, labels in val_loader:  # 검증 데이터 로드
            # print(inputs.shape())
            inputs = inputs.float().to(device)
            labels = labels.float().to(device)
            outputs = model(inputs)  # 모델 예측
            predicted = (outputs >= 0.5).float()  # 확률을 이진 분류로 변환
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # 정확도 출력
    accuracy = correct / total
    print(f"Validation Accuracy: {accuracy:.2%}")

    # 정확도가 이전 최고 정확도보다 높으면 모델 저장
    if accuracy > best_accuracy:
        print("Saving the best model...")
        torch.save(model.state_dict(), model_save_path)
        best_accuracy = accuracy

print("Training complete.")
