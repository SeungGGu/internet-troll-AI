import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# 1. 데이터 불러오기
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
df = pd.read_csv("Dataset.csv", sep="\t")

# 2. 데이터 전처리
# 레이블이 없는 행 처리
null_idx = df[df.lable.isnull()].index
df.loc[null_idx, "lable"] = df.loc[null_idx, "content"].apply(lambda x: x[-1])
df.loc[null_idx, "content"] = df.loc[null_idx, "content"].apply(lambda x: x[:-2])
df = df.astype({"lable": "int"})

# 3. 데이터 분할
train_data = df.sample(frac=0.8, random_state=42)
test_data = df.drop(train_data.index)

# 4. 중복 데이터 제거
train_data.drop_duplicates(subset=["content"], inplace=True)
test_data.drop_duplicates(subset=["content"], inplace=True)

# 5. 모델 및 토크나이저 설정
MODEL_NAME = "beomi/KcELECTRA-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 6. 데이터 토큰화 및 Dataset 생성
tokenized_train_sentences = tokenizer(
    list(train_data["content"]),
    return_tensors="pt",
    max_length=128,
    padding=True,
    truncation=True,
    add_special_tokens=True,
)

tokenized_test_sentences = tokenizer(
    list(test_data["content"]),
    return_tensors="pt",
    max_length=128,
    padding=True,
    truncation=True,
    add_special_tokens=True,
    )

class CurseDataset(torch.utils.data.Dataset):
    # 7. 커스텀 Dataset 정의
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx]).long()
        return item

    def __len__(self):
        return len(self.labels)

# 8. 모델 로딩 및 학습 설정
train_label = train_data["lable"].values.astype('long')
test_label = test_data["lable"].values.astype('long')

train_dataset = CurseDataset(tokenized_train_sentences, train_label)
test_dataset = CurseDataset(tokenized_test_sentences, test_label)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.to(device)

training_args = TrainingArguments(
    output_dir='./',
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=64,
    logging_dir='./logs',
    logging_steps=500,
    save_total_limit=3,
)

# 9. 평가 메트릭 함수 정의
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# 10. Trainer 생성 및 학습 실행 준비
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset, 
    compute_metrics=compute_metrics,
)

# 11. 학습 실행
trainer.train()

# 12. 학습 결과 평가
trainer.evaluate(eval_dataset=test_dataset)

# 13. 모델 저장
trainer.save_model("./path/to/saved/model")
tokenizer.save_pretrained("./path/to/saved/tokenizer")