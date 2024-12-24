import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def valid_label(label):
    if label == 0:
        return '여성/가족'
    elif label == 1:
        return '남성'
    elif label == 2:
        return '성소수자'
    elif label == 3:
        return '인종/국적'
    elif label == 4:
        return '연령'
    elif label == 5:
        return '지역'
    elif label == 6:
        return '종교'
    elif label == 7:
        return '기타 혐오'
    elif label == 8:
        return '악플/욕설'
    elif label == 9:
        return '개인지칭'
    else:
        return '일반문장'

# 모델 아키텍처 생성
loaded_model = AutoModelForSequenceClassification.from_pretrained("beomi/kcbert-base", num_labels=11)

# 저장된 가중치 불러오기
model_save_path = "kc_bert_emotion_classifier.pth"

# 컴퓨터 사용 방식이 gpu일때
# loaded_model.load_state_dict(torch.load(model_save_path))

# 컴퓨터 사용 방식이 cpu일때
loaded_model.load_state_dict(torch.load(model_save_path, map_location=torch.device('cpu')))

# 모델을 평가 모드로 설정
loaded_model.eval()

# 사용자로부터 입력 받기
input_data = input("문장을 입력하세요: ")

tokenizer = AutoTokenizer.from_pretrained("beomi/kcbert-base")
input_encodings = tokenizer(input_data, padding=True, truncation=True, return_tensors="pt")

# 모델에 입력 데이터 전달
with torch.no_grad():
    output = loaded_model(**input_encodings)

# 예측 결과 확인
logits = output.logits
predicted_label = logits.argmax(dim=1).item()

# 예측 결과 출력
predicted_label_str = valid_label(predicted_label)
print(f"입력한 문장: {input_data}")
print(f"예측된 라벨: {predicted_label_str}")
