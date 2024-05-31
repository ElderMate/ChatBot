import os

keyfile_name = r'Key.txt'
key_data = open(keyfile_name, 'r', encoding="utf8")
os.environ["OPENAI_API_KEY"] = key_data.read()

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
    llm = ChatOpenAI(temperature=0, model='gpt-4-turbo-2024-04-09')
    agent = create_json_agent(llm=llm, toolkit=toolkit, max_iterations=1000, prefix=prefix, verbose=True,
                              handle_parsing_errors=True)
    return agent.run(question)


def chatbot_make_answer(question):
    prefix = """
    You are a Korean consultant who informs an elderly person about financial incidents based on all financial-related text messages sent to them. If there is a financial incident the elderly person is unaware of, you must identify the text message where the issue occurred and the nature of the problem. Then, respond to the elderly person that we have checked the problem and will help them handle it later.

    Responses must be provided in JSON format to be used in code. The response format is as follows:
    - If there is no problem: 'answer': 'Response to the elderly person’s statement', 'problem' : 'null'
    - If there is a problem: 'answer': 'Response to the elderly person’s statement', 'problem': ['messageId', 'issue description']
    """

    answer = chatbot_run(prefix, question)
    return answer


def chatbot(question, filename):
    # JSON 데이터 불러 오기 (메시지 정보 불러오기)
    msg_path = r'datafile/msgdata/' + filename + r'.json'
    log_path = r'datafile/logdata/' + filename + r'.json'

    with open(msg_path, 'r', encoding='utf-8') as f1:
        msg_data = json.load(f1)
        f1.close()

    global data
    data = msg_data


    # 다음 시나리오와 답변 생성
    chat_answer = chatbot_make_answer(question)
    json_data = json.loads(chat_answer)

    answer = json_data['answer']
    problem = json_data['problem']



    #문제 발생 시 기록
    if problem != "null":
        print(str(problem[0])+problem[1])
        save_log_data(log_path, problem)

    #대답 반환
    return_data = {
        "answer": answer
    }
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