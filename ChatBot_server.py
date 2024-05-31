import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
import ChatBot, DailyReport
import json
from datafile.datamodel import PromptResponse, makeMsgData, ProgressModel, endModel, DataModel, PromptStartRequest

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
