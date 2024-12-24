import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = torch.device('cpu')

# 1. 저장된 모델 및 토크나이저 로드
model_path = "./path/to/saved/model"
tokenizer_path = "./path/to/saved/tokenizer"

model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# 2. 문장 예측 함수 정의
def sentence_predict(sent):
    # 3. 평가 모드로 변경
    model.eval()

    # 4. 입력 문장 토큰화
    tokenized_sent = tokenizer(
        sent,
        return_tensors="pt",
        truncation=True,
        add_special_tokens=True,
        max_length=128
    )

    # 5. GPU로 모델 이동
    tokenized_sent.to(device)

    # 6. 예측
    with torch.no_grad():
        outputs = model(
            input_ids=tokenized_sent["input_ids"],
            attention_mask=tokenized_sent["attention_mask"],
            token_type_ids=tokenized_sent["token_type_ids"]
        )

    # 7. 결과 반환
    logits = outputs[0]
    logits = logits.detach().cpu()
    result = logits.argmax(-1)
    if result == 0:
        result = " >> 악의적인 댓글 👿"
    elif result == 1:
        result = " >> 일반 댓글 😀"
    return result

# 8. 사용자 입력에 대한 루프
while True:
    sentence = input("댓글을 입력하세요 (0을 입력하면 종료): ")
    if sentence == "0":
        break
    print(sentence_predict(sentence))
    print("\n")

