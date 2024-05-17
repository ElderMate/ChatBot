import os

keyfile_name = r'Key.txt'
key_data = open(keyfile_name, 'r', encoding="utf8")
os.environ["OPENAI_API_KEY"] = key_data.read()

from langchain.agents import create_json_agent
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import JsonToolkit
from langchain.tools.json.tool import JsonSpec

import json

scenario_path = r'datafile/promptdata/scenario.json'

data = {}


def chatbot_run(prompt, question):
    # 툴킷 설정
    spec = JsonSpec(dict_=data, max_value_length=4000)
    toolkit = JsonToolkit(spec=spec)

    # 입력된 프롬프트
    prefix = prompt

    # llm 설정
    llm = ChatOpenAI(temperature=0.1, model='gpt-4-turbo-2024-04-09')
    agent = create_json_agent(llm=llm, toolkit=toolkit, max_iterations=1000, prefix=prefix, verbose=True, handle_parsing_errors=True)
    return agent.run(question)

def chatbot_select_scenario(now_scenario, question):
    prefix = ("I'll give you the current scenario num information and the user's question. "
             + "first, Find the information of the current scenario in scenario data, "
             + "second, Check the current scenario information and let me know the next option according to the user's question. "
             + "Just answer the num information. "
             + "Additionally, The scenario that starts with 3 is the same as 1, so think of it as 1. And if the user's question is trying to end the conversation, you can answer 0. "
             + "Current Scenario: " + now_scenario +", User's Questions: ")
    choice = chatbot_run(prefix, question)
    return choice


def chatbot_make_answer(now_scenario, question):
    prefix = ("You are a Korean chatbot in chat progressing. I will give you the number information of the current scenario and the user's question. Please follow all the instructions : "
              + "First, Take and use your last answer from data in key = chatlog. Just find data[chatlog][chatN](N is num) biggest N is last chatlog"
              + "Second, Use the current scenario number to find the scenario information in the data in key = scenario and refer to it. Remember!, this is an example. Please don't use it. "
              + "Third, using real user msg information from data in key = msg. "
              + "Finally, please make an answer to the user's question based on your last answer and real user data. "
              + "Additionally, if the current scenario starts with 3, you just answer that the message is classified as dangerous message. "
              + "Current Scenario: " + now_scenario +", User's Questions: ")
    answer = chatbot_run(prefix, question)
    return answer

def chatbot_check_message() :
    prefix = (""
              + "Check the all chat logs and pick out the SMS that the user questions did not know "
              + "And, answer picked SMSs in json structure with two keys (id, reason). "
              + "key id is messageId in user SMS data, "
              + "and key reason is why pick out the sms that the user said he did not know. ")
    answer = chatbot_run(prefix, '')
    return answer


def add_chat_log(question, answer):
    global data

    num = len(data['chatlog'])

    chat = {
        "question": question,
        "answer": answer
    }

    data['chatlog']['chat' + str(num)] = chat

def save_chat_log(chat_path):

    with open(chat_path, 'w', encoding='utf-8') as file:
        json.dump(data['chatlog'], file, ensure_ascii=False, indent=4)



def chatbot(now_scenario, question, userid):
    # JSON 데이터 불러 오기
    msg_path = r'datafile/msgdata/' + userid + r'.json'
    chat_path = r'datafile/chatdata/' + userid + r'.json'
    global data
    data = makedata(msg_path, scenario_path, chat_path)

    # 다음 시나리오와 답변 생성
    next_scenario = chatbot_select_scenario(now_scenario, question)

    #시나리오 종료시
    if next_scenario == '0':
        answer = "감사합니다! 오늘도 좋은 하루 되세요!"
        msg_problem = chatbot_check_message()


        with open('test.json', 'w', encoding='utf-8') as file:
            json.dump(json.loads(msg_problem), file, ensure_ascii=False, indent=4)

        return_data = {
            "userid": userid,
            "scenario": "0",
            "answer": answer
        }
        return return_data


    #시나리오 계속 진행
    else :
        answer = chatbot_make_answer(next_scenario, question)

        # 챗 로그 기록
        add_chat_log(question, answer)
        save_chat_log(chat_path)

        # json 형태로 조립
        return_data = {
            "userid": userid,
            "scenario": next_scenario,
            "answer": answer
        }

        #유저한테 응답 보내기
        return return_data


def makedata(msg_path, scen_path, chat_path):
    # 문자 파일 가져오기
    with open(msg_path, 'r', encoding='utf-8') as f1:
        msg_data = json.load(f1)
        f1.close()

    # 시나리오 파일 가져오기
    with open(scen_path, 'r', encoding='utf-8') as f2:
        scenario_data = json.load(f2)
        f2.close()

    # Chat 로그 불러오기
    try:
        f2 = open(chat_path, 'r', encoding='utf-8')
    except FileNotFoundError:
        chat_data = {}
    else:
        chat_data = json.load(f2)
        f2.close()

    # 두 파일 합치기
    make_data = {
        "msg": msg_data,
        "scenario": scenario_data,
        "chatlog": chat_data
    }
    return make_data




