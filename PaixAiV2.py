import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW

class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]

        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'label': label}
    
def check_label(label):
    if label['여성/가족'] == 1:
        return 0
    elif label['남성'] == 1:
        return 1
    elif label['성소수자'] == 1:
        return 2
    elif label['인종/국적'] == 1:
        return 3
    elif label['연령'] == 1:
        return 4
    elif label['지역'] == 1:
        return 5
    elif label['종교'] == 1:
        return 6
    elif label['기타 혐오'] == 1:
        return 7
    elif label['악플/욕설'] == 1:
        return 8
    elif label['개인지칭'] == 1:
        return 9
    else:
        return 10
    
# 혐오 표현 데이터셋
rawdata = pd.read_csv('unsmile_train_v1.0.tsv', sep='\t')

texts = []
labels = []

for idx, label in rawdata.iterrows():
    texts.append(label['문장'])
    labels.append(check_label(label))

# KcBERT 모델과 토크나이저 로드
model_name = "beomi/kcbert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=11)

# 원하는 최대 시퀀스 길이
max_length = 128

labels = torch.tensor(labels, dtype=torch.long)
dataset = CustomDataset(texts, labels, tokenizer, max_length)

# 데이터 로더 생성
batch_size = 11

# Train / Test set 분리
from sklearn.model_selection import train_test_split
train, test = train_test_split(dataset, test_size=0.15, random_state=42)

train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(test, batch_size=batch_size, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 하이퍼파라미터 설정
learning_rate = 1e-5
epochs = 5

# 옵티마이저 및 손실 함수 설정
optimizer = AdamW(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# 모델 재학습
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in train_dataloader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        # 그래디언트 초기화
        optimizer.zero_grad()
        # 모델에 입력을 주어 예측을 생성합니다.
        outputs = model(input_ids, attention_mask=attention_mask)
        # 모델 출력에서 로짓(분류에 대한 점수)을 얻습니다.
        logits = outputs.logits
        # 손실을 계산합니다.
        loss = criterion(logits, labels)
        # 역전파를 통해 그래디언트 계산
        loss.backward()
        # 옵티마이저를 사용해 가중치를 업데이트
        optimizer.step()
        # 에포크 전체 손실을 누적합니다.
        total_loss += loss.item()

    # 에포크 평균 손실 계산
    avg_loss = total_loss / len(train_dataloader)
    # 에포크별 손실 출력
    print(f"Epoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.4f}")

    # 모델 평가
    model.eval()
    val_total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for val_batch in valid_dataloader:
            # Validation 데이터 가져오기
            val_input_ids = val_batch['input_ids']
            val_attention_mask = val_batch['attention_mask']
            val_labels = val_batch['label']

            val_input_ids = val_input_ids.to(device)
            val_attention_mask = val_attention_mask.to(device)
            val_labels = val_labels.to(device)

            # 모델 예측
            val_outputs = model(val_input_ids, attention_mask=val_attention_mask)
            val_logits = val_outputs.logits

            # 손실 계산
            val_loss = criterion(val_logits, val_labels)
            val_total_loss += val_loss.item()

            # 정확도 계산
            val_preds = val_logits.argmax(dim=1)
            correct += (val_preds == val_labels).sum().item()
            total += val_labels.size(0)

    val_avg_loss = val_total_loss / len(valid_dataloader)
    val_accuracy = correct / total
    print(f"Validation Loss: {val_avg_loss:.4f} - Validation Accuracy: {val_accuracy:.4f}")

    # 모델 저장
model_save_path = "kc_bert_emotion_classifier.pth"
torch.save(model.state_dict(), model_save_path)