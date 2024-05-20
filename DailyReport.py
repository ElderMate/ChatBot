from langchain.agents import create_json_agent
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import JsonToolkit
from langchain.tools.json.tool import JsonSpec
import json

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory


import os

keyfile_name = r'Key.txt'
key_data = open(keyfile_name, 'r', encoding="utf8")
os.environ["OPENAI_API_KEY"] = key_data.read()

promptfile_path = r'datafile/promptdata/Dailyprompt.txt'


def makedailyreport(userid, msg_data):

    msg_path = r'datafile/msgdata/' + userid + r'.json'

    #메시지 데이터 저장
    with open(msg_path, 'w', encoding='utf-8') as file:
        json.dump(msg_data, file, ensure_ascii=False, indent=4)
        file.close()
    #프롬프트 읽어오기
    with open(promptfile_path, 'r', encoding='utf-8') as file2:
        prompt = file2.read()
        file2.close()

    #데일리 리포트 생성
    answer = makereport_run(prompt, msg_path)


    return answer

def makereport_run(prompt, msg_path):
    # 문자 파일 가져오기
    file = msg_path
    with open(file, 'r', encoding='utf-8') as f1:
        data = json.load(f1)
        f1.close()

    # 툴킷 설정
    spec = JsonSpec(dict_=data, max_value_length=4000)
    toolkit = JsonToolkit(spec=spec)

    # llm 설정
    llm = ChatOpenAI(temperature=0.2, model='gpt-4-turbo-2024-04-09')
    agent = create_json_agent(llm=llm, toolkit=toolkit, max_iterations=3000, max_execution_time=3000, verbose=True)
    return agent.run(prompt)

