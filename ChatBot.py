import os

keyfile_name = r'Key.txt'
key_data = open(keyfile_name, 'r', encoding="utf8")
os.environ["OPENAI_API_KEY"] = key_data.read().strip()

from langchain.agents import create_json_agent
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import JsonToolkit
from langchain.tools.json.tool import JsonSpec

import json
import requests

data = {}

def chatbot_run(prompt, question):
    # 툴킷 설정
    spec = JsonSpec(dict_=data, max_value_length=4000)
    toolkit = JsonToolkit(spec=spec)

    # 입력된 프롬프트
    prefix = prompt



    # llm 설정
    llm = ChatOpenAI(temperature=0.5, model='gpt-4-turbo-2024-04-09')
    agent = create_json_agent(llm=llm, toolkit=toolkit, max_iterations=1000, prefix=prefix, verbose=True,
                              handle_parsing_errors=True)
    return agent.run(question)


def chatbot_make_answer(question):
    prefix = """
    You are a Korean counselor who conducts financial counseling based on the financial message information of the elderly. Based on the elderly's questions, you can make an answer by referring to the Message Data and ChatLog. in addition, if the elderly answered that they do not know the text message, you write down the "messageId" and the reason they actually don't know the message. And you just answer like : "문제가 확인되었습니다.".
    
    Responses must be provided in JSON format to be used in code. The response format is as follows:
    - If there is no problem: 'answer': 'Response to the elderly person’s statement', 'problem' : null
    - If there is a problem: 'answer': 'Response to the elderly person’s statement', 'problem': ['messageId', 'issue description']
	elder's qusetion : 
    """

    answer = chatbot_run(prefix, question)
    return answer


def chatbot(question, filename):
    # JSON 데이터 불러 오기 (메시지 정보 불러오기)
    msg_path = r'datafile/msgdata/' + filename + r'.json'
    log_path = r'datafile/logdata/' + filename + r'.json'
    chatlog_path = r'datafile/chatlog/' + filename + r'.json'

    with open(msg_path, 'r', encoding='utf-8') as f1:
        msg_data = json.load(f1)
        f1.close()

    try:
        with open(chatlog_path, 'r', encoding='utf-8') as f:
            chatlog_data = json.load(f)
    except FileNotFoundError:
        chatlog_data = {"chats": []}

    global data
    data = {
        "MsgData" : msg_data,
        "ChatLog" : chatlog_data
    }


    #답변 생성
    chat_answer = chatbot_make_answer(question)
    json_data = json.loads(chat_answer)

    answer = json_data['answer']
    problem = json_data['problem']


    #문제 발생 시 기록
    if problem is not None:
        print(str(problem[0])+problem[1])
        save_log_data(log_path, problem)

    #대답 반환
    return_data = {
        "answer": answer
    }

    save_chat_log_data(chatlog_path, question, answer)

    return return_data

def save_log_data(log_path, problem):
    try:
        f2 = open(log_path, 'r', encoding='utf-8')
    except FileNotFoundError:
        log_data = {
            "messageIds": [],
            "reasons": []
        }
    else:
        log_data = json.load(f2)
        f2.close()

    log_data["messageIds"].append(problem[0])
    log_data["reasons"].append(problem[1])


    with open(log_path, 'w', encoding='utf-8') as file:
        json.dump(log_data, file, ensure_ascii=False, indent=4)

def save_chat_log_data(chatlogpath, question, answer):
    try:
        # 파일을 읽어옵니다.
        with open(chatlogpath, 'r', encoding='utf-8') as f:
            chat_log = json.load(f)
    except FileNotFoundError:
        # 파일이 없을 경우 새로운 로그 딕셔너리를 생성합니다.
        chat_log = {"chats": []}

    # 질문과 답변을 로그에 추가합니다.
    chat_log["chats"].append(question)
    chat_log["chats"].append(answer)

    # 로그의 개수가 6개를 초과할 경우, 가장 오래된 질문과 답변 세트를 제거합니다.
    while len(chat_log["chats"]) > 6:
        chat_log["chats"].pop(0)  # 첫 번째 요소(질문) 제거
        chat_log["chats"].pop(0)  # 첫 번째 요소(답변) 제거

    # 수정된 로그를 파일에 다시 씁니다.
    with open(chatlogpath, 'w', encoding='utf-8') as file:
        json.dump(chat_log, file, ensure_ascii=False, indent=4)