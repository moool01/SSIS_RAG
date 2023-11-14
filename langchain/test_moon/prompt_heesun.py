# prompt_template = """
# "다음 정보를 바탕으로 질문을 만드세요.
# ---------------------
# {context_str}
# ---------------------
# 당신은 선생님/교수입니다. 다가오는 퀴즈/시험을 위해 {num_questions_per_chunk}개의 질문을 만드는 것이 당신의 임무입니다. 
# 질문들은 문서에 걸쳐 다양한 성격을 띠어야 하며, 질문은 제공된 문맥 정보에만 제한해야 합니다."
# 제공된 문맥 정보를 바탕으로, 그리고 이전 지식은 사용하지 않고 오직 정보에 기반하여 질문을 생성하십시오.
# """

# anly_prompt_template = """
#     #Problem:
#     다음 정보를 바탕으로 질문을 만드세요.
#     ---------------------
#     {context_str}
#     ---------------------
#     #Instruction:
#     당신은 복지서비스를 추천하고 제공하는 업무를 맡은 굉장히 창의력이 가득하고 열정적인 전문가입니다. 
#     많은 사람이 다양한 문제를 물어보며 간단한 질문에 답하는 것부터 광범위한 주제에 대한 심층적인 토론 및 설명을 제공하는 것까지 다양한 작업을 지원할 수 있습니다.
        
#     현재는 본인이 하고 있는 업무를 데이터화시키는 작업을 수행중이며 복지제도에 대한 설명을 보고 일반 사람들이 검색 할만한 질문을 예상해서 만들고 있습니다. 
#     당신은 창의력이 풍부하고 질문과 답변을 만드는 작업이 재미있어하며 작업이 반복될수록 더 신나서 작업에 몰두합니다.
    
#     질문들은 문서에 걸쳐 다양한 성격을 띠어야 하며, 질문은 제공된 문맥 정보에만 제한해야 합니다.
#     제공된 문맥 정보를 바탕으로, 그리고 이전 지식은 사용하지 않고 오직 정보에 기반하여 일반 사람들이 해당 context와 관련해 복지서비스를 검색할 만한 query를 생성하십시오.
#     {num_questions_per_chunk}개의 질문을 만드는 것이 당신의 임무입니다.
#     ___________________________________________________________________________________________________________________________________________________________
#     #Recall:
    

# """

# prompt_template = """
        # "다음 정보를 바탕으로 질문을 만드세요.
        # ---------------------
        # {context_str}
        # ---------------------
        # 당신은 복지서비스를 추천하고 제공하는 업무를 맡은 굉장히 창의력이 가득하고 열정적인 전문가입니다. 
        # 많은 사람이 다양한 문제를 물어보며 간단한 질문에 답하는 것부터 광범위한 주제에 대한 심층적인 토론 및 설명을 제공하는 것까지 다양한 작업을 지원할 수 있습니다.
            
        # 현재는 본인이 하고 있는 업무를 데이터화시키는 작업을 수행중이며 복지제도에 대한 설명을 보고 일반 사람들이 검색 할만한 질문을 예상해서 만들고 있습니다. 
        # 당신은 창의력이 풍부하고 질문과 답변을 만드는 작업이 재미있어하며 작업이 반복될수록 더 신나서 작업에 몰두합니다.
        
        # 질문들은 문서에 걸쳐 다양한 성격을 띠어야 하며, 질문은 제공된 문맥 정보에만 제한해야 합니다."
        # 제공된 문맥 정보를 바탕으로, 그리고 이전 지식은 사용하지 않고 오직 정보에 기반하여 일반 사람들이 해당 context와 관련해 복지서비스를 검색할 만한 query를 생성하십시오.
        # {num_questions_per_chunk}개의 질문을 만드는 것이 당신의 임무입니다.
        # """

        # query = prompt_template.format(context_str=text, num_questions_per_chunk=num_questions_per_chunk)

        # response = llm.create(
        #     model="gpt-4-1106-preview",
        #     temperature=1,
        #     messages=[{"role":"system","content":query}],
        # )

from langchain.prompts.pipeline import PipelinePromptTemplate
from langchain.prompts.prompt import PromptTemplate

def format_prompt(context:str, num_questions_per_chunk:int)->str:
    full_template = """
    #Problem : {instruction}
    ==========
    #Recall : {recall}
    #Explain your reasoning : {explain}
    ==========
    {answer}
    """

    instruction_template = """
    당신은 복지 사이트에서 복지 제도를 검색하는 사용자입니다. 복지 제도를 인터넷에서 검색 한다고 했을 때, {num_questions_per_chunk}개의 질문을 검색하는 것이 당신이 할 일입니다.
    질문들은 전체 문서에 걸쳐서 다양한 내용을 포함해야 하고, 질문은 제공된 문맥 정보로만 제한해야 합니다.
    제공된 문맥 정보를 바탕으로, 그리고 이전 지식은 사용하지 않고 오직 정보에 기반하여 질문을 생성하십시오.
    ==========
    #Context : 
    {context}"""
    instruction_prompt = PromptTemplate.from_template(instruction_template)

    recall_template = """문제와 관련한 예시 3개를 떠올려 보세요. 각각의 문제는 논리적인 구조로 생성해야 합니다."""
    recall_prompt = PromptTemplate.from_template(recall_template)

    explain_template = """ 생성한 이유를 생각해보세요. 이유의 형식은 다음과 같습니다 이유는 작성하면 안됩니다. . 
    - 첫 번째로, 저는 이렇게 생각했습니다 ... 
    - 다음으로, 저는 이렇게 생각했습니다 ...
    - 결론적으로, 저는 이렇게 생각했습니다 ... 
    이유는 대답에 작성하면 안됩니다. """
    explain_prompt = PromptTemplate.from_template(explain_template)

    answer_template = """앞선 이유를 바탕으로, 질문을 생성해주세요. 다른 내용 없이 생성한 질문만 작성하면 됩니다. 

    {num_questions_per_chunk}개의 질문을 생성하면 다음과 같습니다 ... """
    answer_prompt = PromptTemplate.from_template(answer_template)

    input_prompts = [
        ("instruction", instruction_prompt),
        ("recall", recall_prompt),
        ("explain", explain_prompt),
        ("answer", answer_prompt),
    ]
    prompt = PromptTemplate.from_template(full_template)
    prompt_pipeline = PipelinePromptTemplate(final_prompt=prompt, pipeline_prompts=input_prompts)

    # print(prompt_pipeline)
    # prompt_pipeline.input_variables

    final = prompt_pipeline.format(
        context = context,
        num_questions_per_chunk = num_questions_per_chunk
    )
    return final

