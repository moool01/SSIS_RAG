from langchain.prompts.pipeline import PipelinePromptTemplate
from langchain.prompts.prompt import PromptTemplate

###### form pydantic output parser
from langchain.pydantic_v1 import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser

from typing import List

def _get_pydantic():
    """ pydantic 형태로 결과 parsing해서 돌려줌. -> output 구조 설정과 결과 parsing 2번 사용해야 함. """
    class analogical_thinking(BaseModel):
        reasoning: str|List[str] = Field(
        description="논리적 구조를 나타내는 문자열 목록. 각 문자열은 특정 아날로그 추론 시나리오를 설명한다."
        )
        questions: str|List[str] = Field(
        description="생성된 질문들. 각 질문은 해당 추론 시나리오에 맞게 구성된다."
        )

    output = PydanticOutputParser(pydantic_object=analogical_thinking)
    return output

##### example
def da_format_prompt(context:str, num_questions_per_chunk:int)->str:

    # 전체 구조
    full_template = """
    #Problem : {instruction}
    ==========
    #Recall : {recall}
    #Explain your reasoning : {explain}
    ==========
    {answer}
    """

    # 전체 구조에서 {instruction}에 해당하는 부분. num_questions_per_chunk, context는 input variable로서 작동.
    instruction_template = """
    당신은 복지 사이트에서 복지 제도를 검색하는 사용자입니다. 복지 제도를 인터넷에서 검색 한다고 했을 때, {num_questions_per_chunk}개의 질문을 검색하는 것이 당신이 할 일입니다.
    질문들은 전체 문서에 걸쳐서 다양한 내용을 포함해야 하고, 질문은 제공된 문맥 정보로만 제한해야 합니다.
    제공된 문맥 정보를 바탕으로, 그리고 이전 지식은 사용하지 않고 오직 정보에 기반하여 질문을 생성하십시오.
    ==========
    #Context : 
    {context}"""
    instruction_prompt = PromptTemplate.from_template(instruction_template)

    # 전체 구조에서 {recall}에 해당하는 부분 -> Analogical Prompting에서 LLM이 문서를 보고 생성한 예시
    recall_template = """주어진 정보를 바탕으로 문서에 궁금한 점을 5가지 정도 떠올려 보세요 ... """
    recall_prompt = PromptTemplate.from_template(recall_template)

    # 전체 구조에서 {explain}에 해당하는 부분 -> recall에 대한 이유를 논리적 구조에 따라 생성함
    explain_template = """ 예시에 대해 논리적으로 생각한 후에 이유를 작성해보세요. 형식은 해당 사항을 준수해서 작성하세요. 
    - 첫 번째로, 저는 이렇게 생각했습니다 ... 
    - 다음으로, 저는 이렇게 생각했습니다 ...
    - 결론적으로, 저는 이렇게 생각했습니다 ... """
    explain_prompt = PromptTemplate.from_template(explain_template)

    # 전체 구조에서 {answer}에 해당하는 부분 -> 앞의 recall과 explain을 보고, LLM에서 num_questions_per_chunk의 개수에 해당하는 문제의 수만큼 돌려줌. format_instruction은 pydantic object.
    answer_template = """앞선 예시와 비슷하게, {num_questions_per_chunk}개의 질문을 생성해주세요. 다른 내용 없이 생성한 질문만 작성하면 됩니다. 질문은 무조건 한국어로 작성해주세요.
    주어진 정보 전체에 따라 {num_questions_per_chunk}개의 질문을 생성하는 구조는 다음과 같습니다.

    {format_instruction}
    ... """
    answer_prompt = PromptTemplate.from_template(answer_template)

    # 전체 프롬프트 구성
    input_prompts = [
        ("instruction", instruction_prompt),
        ("recall", recall_prompt),
        ("explain", explain_prompt),
        ("answer", answer_prompt),
    ]
    prompt = PromptTemplate.from_template(full_template)
    prompt_pipeline = PipelinePromptTemplate(final_prompt=prompt, pipeline_prompts=input_prompts)

    # 전체 프롬프트를 구성하고, input variable에 해당하는 변수 넣어서 LLM에 넣을 최종 내용으로 작성한다.
    pydantic = _get_pydantic()

    #prompt config
    final = prompt_pipeline.format(
        context = context,
        num_questions_per_chunk = num_questions_per_chunk,
        format_instruction = pydantic.get_format_instructions(),
    )
    return final #<class 'langchain.prompts.base.StringPromptValue'>

    
# print(type(da_format_prompt(context="안녕",num_questions_per_chunk=2)))





