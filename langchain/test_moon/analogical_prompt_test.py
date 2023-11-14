import os

from langchain.prompts.pipeline import PipelinePromptTemplate
from langchain.prompts import PromptTemplate

from langchain.chains import RetrievalQA

from settings import openai_api_key
from openai import OpenAI

os.environ["OPENAI_API_KEY"] = openai_api_key

prompt_template = """
#Problem: {instruction}
==========
#Recall : {recall}
#Explain your reasoning : {explain}
==========
#Question: {query}
"""

instruction_template = """
당신은 지역사회에서 다양한 층의 사람들을 돕고 지원하는 사회복지사입니다.
여러 사회문제에 대한 인식과 사회적 책임감을 가지고 있으며, 개별 개인의 상황을 정확히 파악하여 맞춤형 도움을 제공하는 데 주력하고 있습니다.
사회복지사로서 아래 {context}를 기반으로 하여, 사람들의 질문에 최대한 상세하고 정확한 정보를 제공하기 위해 노력하고 있습니다.
==========
#Context : 
{context}"""
instruction_prompt = PromptTemplate.from_template(instruction_template)

recall_template = """문제와 관련한 예시 3개를 떠올려 보세요. 각각의 문제는 논리적인 구조로 생성해야 합니다."""
recall_prompt = PromptTemplate.from_template(recall_template)

explain_template = """ 생성한 이유를 생각해보세요. 이유의 형식은 다음과 같습니다 이유는 작성하면 안됩니다.
- 첫 번째로, 저는 이렇게 생각했습니다 ... 
- 다음으로, 저는 이렇게 생각했습니다 ...
- 결론적으로, 저는 이렇게 생각했습니다 ... 
이유는 답변에 작성하면 안됩니다. """
explain_prompt = PromptTemplate.from_template(explain_template)

answer_template = """앞선 {recall}을 바탕으로, 답변을 생성해주세요.
다른 내용 없이 생성한 답변만 작성하면 됩니다."""
answer_prompt = PromptTemplate.from_template(answer_template)

input_prompts = [
        ("instruction", instruction_prompt),
        ("recall", recall_prompt),
        ("explain", explain_prompt),
    ]
        
prompt = PromptTemplate.from_template(prompt_template)
prompt_pipeline = PipelinePromptTemplate(final_prompt=prompt, pipeline_prompts=input_prompts)

def lcel_prompt(retriever):
    final = prompt_pipeline.format(
    context=retriever,
    query='가사, 간병과 관련하여 도움이 필요한데 어떤 도움을 받을 수 있나요? 조건도 알려주세요.'
    )
    
# response = llm.create(
#     model = "gpt-3.5-turbo-1106",
#     temperature=0
#     messages=[{"role":"system","content":final}],
#     )

# response