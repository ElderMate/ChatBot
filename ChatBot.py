import os

keyfile_name = r'Key.txt'
key_data = open(keyfile_name, 'r', encoding="utf8")
os.environ["OPENAI_API_KEY"] = key_data.read()

from langchain.agents import create_json_agent
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import JsonToolkit
from langchain.tools.json.tool import JsonSpec

import json

data = {}

def chatbot_run(prompt, question):
    # 툴킷 설정
    spec = JsonSpec(dict_=data, max_value_length=4000)
    toolkit = JsonToolkit(spec=spec)

    # 입력된 프롬프트
    prefix = prompt

    # llm 설정
    llm = ChatOpenAI(temperature=0.1, model='gpt-4-turbo-2024-04-09')
    agent = create_json_agent(llm=llm, toolkit=toolkit, max_iterations=1000, prefix=prefix, verbose=True,
                              handle_parsing_errors=True)
    return agent.run(question)


def chatbot_make_answer(question):
    prefix = ("너는 노인을 위한 금융 상담사야. 질문에 대해서 적절한 답변을 한국어로 만들어줘."
              + " 그리고 노인이 모르는 금융 내역이 있다면 그 메시지의 id 정보를 함께 알려줘"
              + " 내가 처리할 수 있게 json 형태의 답변으로 해줘. 형태는 'answer' : '답변', 'problem' : 'messageId' 이것만 답변해줘."
              + " problem은 null로 해주다가, 노인이 모른다고 대답하면 그 때 적어줘"
              + " 추가적으로 노인이 상담을 종료하는 말을 한다면 answer에 '0'을 넣어서 답변해줘"
              + " 노인의 질문: ")
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
    problem_id = json_data['problem']


    # 시나리오 종료시 서버로 종료 요청
    if answer == '0':

        #데이터 생성
        with open(log_path, 'r', encoding='utf-8') as f1:
            log_data = json.load(f1)
            f1.close()

        request_data = {
            "filename": filename,
            "problems": log_data
        }

        #서버로 종료 요청 전송
        print(request_data)

        #App으로 끝 반환
        return_data = {
            "answer": "0"
        }

        return return_data

    # 시나리오 계속 진행
    else:
        #문제 발생 시 기록
        if problem_id != "null":
            save_log_data(log_path, problem_id, "")

        #대답 반환
        return_data = {
            "answer": answer
        }
        return return_data

def save_log_data(log_path, msg_id, reason):
    try:
        f2 = open(log_path, 'r', encoding='utf-8')
    except FileNotFoundError:
        log_data = {}
    else:
        log_data = json.load(f2)
        f2.close()

    num = len(data)
    chat_problem = {
        "messageId": msg_id,
        "reason": reason
    }
    log_data['problem' + str(num)] = chat_problem

    with open(log_path, 'w', encoding='utf-8') as file:
        json.dump(log_data, file, ensure_ascii=False, indent=4)