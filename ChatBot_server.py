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
    "nonPayments": "미납",
    "opens": "계좌 개설",
    "rejects": "결제 거절"
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
        request_data = request.dict()
        translated_data = translate_keys(request_data, key_mapping)

        #데일리 리포트 생성
        dailyreport = DailyReport.makedailyreport(translated_data["filename"], translated_data)

        #생성값 반환
        return {
            "response": dailyreport,
            "next": 1
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




@app.post("/reports/progress")
async def process_data(model: ProgressModel):
    userid = model.id
    question = model.question
    scenario = model.scenario

    return_data = ChatBot.chatbot(scenario, question, userid)

    return return_data


@app.post("/endAPI")
async def process_data(model: endModel):

    userid = model.id

    os.remove("datafile/chatdata/" + userid + ".json")
    os.remove("datafile/msgdata/" + userid + ".json")
