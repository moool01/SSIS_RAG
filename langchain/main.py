""" Dataset generation Example usage """

from document import BaseDBLoader
from qa_generation import generate_qa 
import os
import json

import time
from datetime import datetime

####### 1. From raw Markdownfiles to corpus -> {uuid:text} dict
# loader = BaseDBLoader()
# document = loader.load(is_regex=True, is_split=False)
# # print(document)
# corpus = loader.get_corpus()
# with open('result.json', 'w', encoding='utf-8') as file :
#     json.dump(corpus, file, indent='\t', ensure_ascii=False)
# print(len(corpus))
# # 464


######## 2. Synthetic Dataset Generating from corpus
# directory = os.path.dirname(__file__)
# os.chdir(directory)

# ### 데이터셋 10개씩 batch 나눠서 진행하였음.
# folder_path = "./batch"
# folder_result = "./resultQA"

# if not os.path.exists(folder_result) :
#     os.makedirs(folder_result)

# for number, filename in enumerate(os.listdir(folder_path), start=1):
#     #make result folder first
#     print("="*60)
#     print (f"dataset generate on batch_{number} start on : {datetime.now()}\n")

#     if filename.endswith('.json'):
#         file_path = os.path.join(folder_path, filename)
#         with open(file_path, 'r', encoding="utf-8") as file:
#             data = json.load(file)
#         with open(f"./resultQA/batch_{number}_result.json", 'w+', encoding="utf-8") as result_file:
#             json.dump(generate_qa(data, num_questions_per_chunk=25), result_file, indent="\t", ensure_ascii=False)
#     time.sleep(3)

### 생성된 Dataset 합치기 -> batch_1~47_result ..... -> qa_dataset.json
# print(len(os.listdir(folder_result))) #47
# merged_batch_dataset = {
#     'queries':{},
#     'corpus':{},
#     'relevant_docs':{},
# }

# for file in sorted(os.listdir(folder_result)):
#     with open(os.path.join(folder_result, file), "r", encoding='utf-8') as batch_file :
#         batch_dict = json.load(batch_file)

#         for key in batch_dict.keys() :
#             merged_batch_dataset[key].update(batch_dict[key])

# with open('./da_qa_dataset.json', 'w+',encoding='utf-8') as qa_dataset :
#     json.dump(merged_batch_dataset, da_qa_dataset, indent='\t', ensure_ascii=False)


    

