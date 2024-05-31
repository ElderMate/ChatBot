import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
import ChatBot, DailyReport
import json
from datafile.datamodel import PromptResponse, makeMsgData, ProgressModel, endModel, DataModel, PromptStartRequest
from pydantic import BaseModel
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch
import joblib
import numpy as np

app = FastAPI()

key_mapping = {
    "fileName": "filename",
    "autoTransfers": "자동 이체",
    "cancels": "결제 취소",
    "confirms": "결제 승인",
    "invoices": "납부 예정",
    "opens": "계좌 개설"
}

def translate_keys(data, mapping):
    """기존 데이터의 키를 새로운 키로 변환하는 함수"""
    if isinstance(data, dict):
        return {mapping.get(k, k): translate_keys(v, mapping) for k, v in data.items()}
    elif isinstance(data, list):
        return [translate_keys(item, mapping) for item in data]
    else:
        return data


@app.post("/reports/start")
async def process_data(request: PromptStartRequest):
    try:
        # 요청을 받아 처리하는 로직
        # 예시: 요청된 데이터를 확인하여 필요한 비즈니스 로직을 수행합니다.
        if not os.path.exists(r'datafile/msgdata') :
            os.makedirs(r'datafile/msgdata')
        if not os.path.exists(r'datafile/logdata') :
            os.makedirs(r'datafile/logdata')
        request_data = request.dict()
        translated_data = translate_keys(request_data, key_mapping)
        print(translated_data)

        #데일리 리포트 생성
        try :
            dailyreport = DailyReport.makedailyreport(translated_data["filename"], translated_data)
            print(dailyreport)
        except Exception as e:
            print(e)

        #생성값 반환
        return {
            "response": dailyreport
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




@app.post("/reports/progress")
async def process_data(model: ProgressModel):

    filename = model.fileName
    question = model.question
    print(model)

    return_data = ChatBot.chatbot(question, filename)

    return return_data


@app.post("/reports/end")
async def process_data(model: endModel):

    try:
        userid = model.fileName
    except Exception as e :
        print(e)

    log_path = r'datafile/logdata/' + userid + r'.json'
    try:
        f2 = open(log_path, 'r', encoding='utf-8')
    except FileNotFoundError:
        log_data = {}
    else:
        log_data = json.load(f2)
        f2.close()


    #파일 삭제
    msg_path = r'datafile/msgdata/' + userid + r'.json'
    log_path = r'datafile/logdata/' + userid + r'.json'
    os.remove(msg_path)
    os.remove(log_path)
    print(log_data)
    return log_data


class Item(BaseModel):
    message: str


# 각 클래스별 모델과 id2tag 설정
models_info = {
    "payment_denial": {
        "model_path": "kobert_결제거절",
        "id2tag": {0: 'O', 1: 'METHOD', 2: 'LOCATION', 3: 'TIME', 4: 'COST', 5: "REJECTION_REASON"}
    },
    "payment_approval": {
        "model_path": "kobert_결제승인",
        "id2tag": {0: 'O', 1: 'METHOD', 2: 'LOCATION', 3: 'TIME', 4: 'COST'}
    },
    "payment_cancellation": {
        "model_path": "kobert_결제취소",
        "id2tag": {0: 'O', 1: 'METHOD', 2: 'LOCATION', 3: 'TIME', 4: 'COST'}
    },
    "account_opening": {
        "model_path": "kobert_계좌개설",
        "id2tag": {0: 'O', 1: 'BANK', 2: 'TYPE'}
    },
    "payment_scheduled": {
        "model_path": "kobert_납부예정",
        "id2tag": {0: 'O', 1: 'PAYEE', 2: 'COST', 3: 'TIME', 4: 'PAYMENT REASON'}
    },
    "non_payment": {
        "model_path": "kobert_미납",
        "id2tag": {0: 'O', 1: 'PAYEE', 2: 'COST', 3: 'TIME'}
    },
    "automatic_transfer": {
        "model_path": "kobert_자동이체",
        "id2tag": {0: 'O', 1: 'BANK', 2: 'COMPANY'}
    }
}


def load_model(model_name):
    model_path = "./model/" + model_name
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return model, tokenizer, device


def preprocess_text(text, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512,
                       return_offsets_mapping=True)
    return inputs


def predict_categories(model, tokenizer, device, inputs):
    model.eval()
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=-1)
    return predictions, inputs['offset_mapping'].detach().cpu().numpy()[0]


def extract_information(predictions, offset_mapping, original_text, id2tag):
    labels = [id2tag[id] for id in predictions[0].cpu().numpy()]
    extracted_info = {tag: "" for tag in id2tag.values() if tag != 'O'}

    for i, (offset, label) in enumerate(zip(offset_mapping, labels)):
        if label != "O":
            start, end = offset
            extracted_text = original_text[start:end]
            if label in extracted_info:
                extracted_info[label] += extracted_text + " "

    for key in extracted_info:
        extracted_info[key] = extracted_info[key].strip()

    return extracted_info


@app.post("/keywords/{class_name}")
async def create_item(class_name: str, item: Item):
    if class_name not in models_info:
        raise HTTPException(status_code=404, detail="Class not found")

    model_info = models_info[class_name]
    model, tokenizer, device = load_model(model_info["model_path"])
    id2tag = model_info["id2tag"]

    try:
        preprocessed_inputs = preprocess_text(item.message, tokenizer)
        predictions, offset_mapping = predict_categories(model, tokenizer, device, preprocessed_inputs)
        extracted_info = extract_information(predictions, offset_mapping, item.message, id2tag)

        return extracted_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# 랜덤 포레스트 모델과 TF-IDF 벡터라이저 로드
model_path = './model/random_forest_model.pkl'
vectorizer_path = './vectorizer/tfidf_vectorizer.joblib'
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

index_to_label = {
    0: '납부 예정',
    1: '결제 거절',
    2: '결제 승인',
    3: '자동 이체',
    4: '미납',
    5: '결제 취소',
    6: '계좌 개설'
}

def get_label_from_index(index):
    return index_to_label.get(index, "Unknown")

def preprocess_text2(text):
    # 텍스트를 TF-IDF 벡터로 변환
    vectorized_text = vectorizer.transform([text])
    return vectorized_text

class TextItem(BaseModel):
    text: str

@app.post("/class/")
async def predict(item: TextItem):
    try:
        preprocessed_text = preprocess_text2(item.text)
        prediction = model.predict(preprocessed_text)[0]
        predicted_index = np.argmax(prediction)
        return {"response": get_label_from_index(predicted_index)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

