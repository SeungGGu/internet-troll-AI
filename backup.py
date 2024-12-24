from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import mysql.connector
from mysql.connector import errorcode
from datetime import datetime

app = Flask(__name__)
CORS(app)

# AI 모델 로드 - 첫 번째 AI (악플 분류)
loaded_model = AutoModelForSequenceClassification.from_pretrained("beomi/kcbert-base", num_labels=11)
model_save_path = "kc_bert_emotion_classifier.pth"
loaded_model.load_state_dict(torch.load(model_save_path, map_location=torch.device('cpu')))
loaded_model.eval()
tokenizer = AutoTokenizer.from_pretrained("beomi/kcbert-base")

# AI 모델 로드 - 두 번째 AI (악플 여부 감지)
device = torch.device('cpu')
model_path = "./path/to/saved/model"
tokenizer_path = "./path/to/saved/tokenizer"
second_model = AutoModelForSequenceClassification.from_pretrained(model_path)
second_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# MySQL 데이터베이스 로드
config = {
    'user': 'root',
    'password': '0000',
    'host': 'localhost',
    'database': 'scv_web',
    'raise_on_warnings': True
}

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
        return '악플'
    
def update_reply_count(article_no):
    try:
        cnx = mysql.connector.connect(**config)
        cursor = cnx.cursor()

        # Update reply count in tbl_article
        update_query = (
            "UPDATE tbl_article "
            "SET reply_cnt = reply_cnt + 1 "
            "WHERE article_no = %s"
        )

        cursor.execute(update_query, (article_no,))
        cnx.commit()
        print("댓글 수 증가 완료")
    except mysql.connector.Error as err:
        print("Error updating reply count:", err)
        # Handle specific errors if needed
    finally:
        cursor.close()
        cnx.close()

def insert_into_tbl_reply(article_no, reply_writer, reply_text, predicted_label_str):
    try:
        cnx = mysql.connector.connect(**config)
        cursor = cnx.cursor()

        # Insert comment information into tbl_reply
        add_comment = (
            "INSERT INTO tbl_reply (article_no, reply_writer, reply_text, reply_ctg) "
            "VALUES (%s, %s, %s, %s)"
        )

        data_comment = (article_no, reply_writer, reply_text, predicted_label_str)

        cursor.execute(add_comment, data_comment)
        cnx.commit()
        print("저장됨")
    except mysql.connector.Error as err:
        print("Error inserting into database:", err)
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("Something is wrong with your user name or password")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("Database does not exist")
        else:
            print(err)
    finally:
        cursor.close()
        cnx.close()

# AI 모델을 사용하여 악플 여부를 검사하는 함수
def check_for_toxicity(text):
    second_model.eval()
    tokenized_text = second_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )
    tokenized_text.to(device)
    with torch.no_grad():
        outputs = second_model(
            input_ids=tokenized_text["input_ids"],
            attention_mask=tokenized_text["attention_mask"],
            token_type_ids=tokenized_text["token_type_ids"]
        )
    logits = outputs.logits
    predicted_label = logits.argmax(dim=1).item()
    if predicted_label == 0:
        print("악플로 감지함")
        return True  # 악플인 경우
    else:
        print("선플로 감지함")
        return False  # 악플이 아닌 경우

@app.route('/add_comment', methods=['POST'])
def add_comment():
    data = request.get_json()   
    article_no = data.get('articleNo')
    reply_writer = data.get('replyWriter')
    reply_text = data.get('replyText')

    # 첫 번째 AI를 사용하여 라벨 예측
    input_encodings = tokenizer(reply_text, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        output = loaded_model(**input_encodings)

    logits = output.logits
    predicted_label = logits.argmax(dim=1).item()
    predicted_label_str = valid_label(predicted_label)

    print("글번호 : " + article_no) 
    print("작성자 : " + reply_writer) 
    print("글내용 : " + reply_text) 
    print("글분류 : " + predicted_label_str) 

    # 악플 여부 확인
    is_toxic = check_for_toxicity(reply_text)

    if not is_toxic:
        # 악플이 아닌 경우 데이터베이스에 저장 (댓글 분류: 일반문장)
        insert_into_tbl_reply(article_no, reply_writer, reply_text, "일반문장")
    else:
        # 악플인 경우 데이터베이스에 저장 (댓글 분류: 첫 번째 AI로 예측된 라벨)
        insert_into_tbl_reply(article_no, reply_writer, reply_text, predicted_label_str)

    # 댓글 수 증가
    update_reply_count(article_no)

    # JSON 형태로 반환
    return jsonify({
        'articleNo': article_no,
        'replyWriter': reply_writer,
        'replyText': reply_text,
        'predictedLabel': predicted_label_str
    })

if __name__ == '__main__':
    app.run(debug=True)
