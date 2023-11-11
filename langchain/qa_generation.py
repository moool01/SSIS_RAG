""" LangChain version QA generation (currently use OpenAI) """

import os
import uuid

from settings import openai_api_key

from tqdm.notebook import tqdm
from openai import OpenAI
from langchain.chat_models.openai import ChatOpenAI
from openai import OpenAI

from prompt import da_format_prompt

from langchain.output_parsers import PydanticOutputParser
from langchain.pydantic_v1 import BaseModel, Field
from langchain.schema.output_parser import OutputParserException
from typing import List

from langchain.output_parsers import RetryWithErrorOutputParser

from langchain.prompts import SystemMessagePromptTemplate



os.environ["OPENAI_API_KEY"] = openai_api_key

def _write_dataset(text:str, node_id:str, questions:list, queries:dict, corpus:dict, relevant_docs:dict)->None:
    for question in questions :
        question_id = str(uuid.uuid4())

        queries[question_id] = question
        corpus[node_id] = text
        relevant_docs[question_id] = [node_id]

##### 1109 기준 OpenAI 최신버전(1.2.0) 있어야 가동 가능.
def generate_qa(context:dict, model:str="gpt-3.5-turbo-1106", num_questions_per_chunk:int=2)->dict:
    dataset = {}
    queries = {}
    corpus = {}
    relevant_docs = {}
    # llm = OpenAI().chat.completions

    llm = OpenAI().chat.completions
    # llm = ChatOpenAI(model=model, temperature=0.7)
    
    #outputparser config
    class analogical_thinking(BaseModel):
        reasoning: str|List[str] = Field(
        description="논리적 구조를 나타내는 문자열 목록. 각 문자열은 특정 아날로그 추론 시나리오를 설명한다."
        )
        questions: str|List[str] = Field(
        description="생성된 질문들. 각 질문은 해당 추론 시나리오에 맞게 구성된다."
        )

    pydanticoutput = PydanticOutputParser(pydantic_object=analogical_thinking)

    #main
    for node_id, text in tqdm(context.items()) :
        query = da_format_prompt(context=text, num_questions_per_chunk=num_questions_per_chunk)
        # llm config(OpenAI API function)
        response = llm.create(
            model = model,
            temperature=0,
            messages=[{"role":"system","content":query}],
            )
        output = response.choices[0].message.content.strip()

        try : 
            parsed_output = pydanticoutput.parse(output)
        except OutputParserException : 
            retry_parser = RetryWithErrorOutputParser.from_llm(parser=pydanticoutput, llm=ChatOpenAI(model=model, temperature=1))
            parsed_output = retry_parser.parse_with_prompt(output, prompt_value=query)

        print(f"""\n\n 논리에 따라 생성한 결과는 다음과 같습니다. 
              {parsed_output.reasoning}\n\n{parsed_output.questions}\n\n.""")

        questions = parsed_output.questions
        # questions=[re.sub(r"^\d+[\).\s]", "", question).strip() for question in parsed_response] #원래 text로 나올 경우 정규식 이용해서 자른 것. (legacy)
        questions=[question for question in questions if len(question) > 0]
        _write_dataset(text=text, node_id=node_id, questions=questions, queries=queries, corpus=corpus, relevant_docs=relevant_docs)

    dataset = {'queries':queries, 'corpus':corpus, 'relevant_docs':relevant_docs}
    return dataset

##test
# generate_qa(context={"1234":"겨울 LPG 난방 지원 사업","11345-123-ab":"참전용사 보훈 지원"}, num_questions_per_chunk=3)

