import json

from pydantic import BaseModel
from typing import List, Optional


class PromptResponse(BaseModel):
    message: str
    success: bool


class AutoTransfer(BaseModel):
    messageId: int
    bank: str
    company: str

class Cancel(BaseModel):
    messageId: int
    method: str
    location: str
    time: str
    cost: str

class Confirm(BaseModel):
    messageId: int
    method: str
    location: str
    time: str
    cost: str

class Invoice(BaseModel):
    messageId: int
    payee: str
    cost: str
    time: str
    paymentReason: Optional[str]

class NonPayment(BaseModel):
    messageId: int
    payee: str
    cost: Optional[str]
    time: str

class Open(BaseModel):
    messageId: int
    bank: str
    type: str

class Reject(BaseModel):
    messageId: int
    method: str
    location: Optional[str]
    time: str
    cost: Optional[str]
    rejectReason: str

# 최종적인 요청 데이터 모델
class PromptStartRequest(BaseModel):
    fileName: str
    autoTransfers: List[AutoTransfer]
    cancels: List[Cancel]
    confirms: List[Confirm]
    invoices: List[Invoice]
    nonPayments: List[NonPayment]
    opens: List[Open]
    rejects: List[Reject]


def makeMsgData(rawdata):
    json_data = {
        "id": rawdata["filename"],
        "message": {
            "결제 승인": rawdata["confirms"],
            "결제 거절": rawdata["rejects"],
            "결제 취소": rawdata["cancels"],
            "계좌 개설": rawdata["opens"],
            "납부 예정": rawdata["invoices"],
            "미납": rawdata["nonPayments"],
            "자동 이체": rawdata["autoTransfers"]
        }
    }

    return json_data


class ProgressModel(BaseModel):
    filename: str
    question: str


class endModel(BaseModel):
    id: str


class DataModel(BaseModel):
    data: str
