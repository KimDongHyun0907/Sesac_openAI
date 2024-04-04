import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# pickle 파일 읽기
import pickle
import gzip

with gzip.open('transfer.pickle', 'rb') as f:
    dataset = pickle.load(f)

# 데이터셋을 학습용과 검증용으로 분리합니다.
train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)

# BERT 토크나이저를 불러옵니다.
tokenizer = BertTokenizer.from_pretrained('sgunderscore/hatescore-korean-hate-speech')

# 모델을 불러옵니다.
model = BertForSequenceClassification.from_pretrained('sgunderscore/hatescore-korean-hate-speech')

# GPU를 사용할 수 있는지 확인합니다.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 모델과 데이터를 GPU로 이동합니다.
model.to(device)

# 데이터를 토큰화하고 인덱스로 변환합니다.
def tokenize_data(data):
    input_ids = []
    labels = []
    for _, data in data.iterrows():
        sentence, label = data['sentence'], data['clean']
        encoded_dict = tokenizer.encode_plus(
                            sentence,
                            add_special_tokens = True,
                            max_length = 128,
                            padding = 'max_length',
                            return_attention_mask = True,
                            return_tensors = 'pt'
                        )
        input_ids.append(encoded_dict['input_ids'])
        labels.append(label)  # 여기서 label을 리스트에 추가해야 합니다.
    
    input_ids = torch.cat(input_ids, dim=0).to(device)  # GPU로 이동
    labels = torch.tensor(labels).to(device)  # GPU로 이동
    return input_ids, labels

# 훈련 데이터와 검증 데이터를 토큰화합니다.
train_input_ids, train_labels = tokenize_data(train_data)
val_input_ids, val_labels = tokenize_data(val_data)

# 배치 사이즈 설정
batch_size = 32

# DataLoader를 사용하여 데이터를 미니배치로 나눕니다.
train_dataloader = DataLoader(
            torch.utils.data.TensorDataset(train_input_ids, train_labels), 
            batch_size=batch_size, 
            shuffle=True
        )
val_dataloader = DataLoader(
            torch.utils.data.TensorDataset(val_input_ids, val_labels), 
            batch_size=batch_size, 
            shuffle=False
        )

# 옵티마이저 및 손실 함수 설정
optimizer = AdamW(model.parameters(), lr=2e-7, eps=1e-8)
loss_fn = torch.nn.CrossEntropyLoss()

from tqdm import tqdm  # tqdm 임포트

# 모델 훈련
epochs = 10
best_val_accuracy = 0.0
best_model_path = "best_model_transfer.pth"  # 모델을 저장할 경로 지정

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(enumerate(train_dataloader), desc=f'Epoch {epoch + 1}/{epochs}', leave=False)  # tqdm 사용

    for step, batch in progress_bar:
        # 배치를 GPU로 이동
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_labels = batch
        
        # Forward pass
        outputs = model(b_input_ids)
        logits = outputs.logits

        # Loss 계산
        loss = loss_fn(logits, b_labels)  # CrossEntropyLoss는 정수 형태의 클래스 인덱스를 받습니다.
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Optimizer로 가중치 업데이트
        optimizer.step()
        
        # 가중치 초기화
        model.zero_grad()
        
        running_loss += loss.item()
        progress_bar.set_postfix({'running_loss': running_loss / (step + 1)})  # running_loss를 평균내어 출력

    # 검증 데이터셋을 이용하여 모델 평가
    model.eval()
    val_predictions = []
    val_labels = []
    with torch.no_grad():
        for batch in val_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_labels = batch
            
            # Forward pass
            outputs = model(b_input_ids)
            logits = outputs.logits
            
            # 예측값 저장
            val_predictions.extend(np.argmax(logits.cpu().detach().numpy(), axis=1))
            val_labels.extend(b_labels.cpu().detach().numpy())
    
    # 정확도 계산
    val_accuracy = accuracy_score(val_labels, val_predictions)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_dataloader)}, Validation Accuracy: {val_accuracy}")

    # 정확도가 높아지면 모델 저장
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), best_model_path)

print('best accuracy', best_val_accuracy)
