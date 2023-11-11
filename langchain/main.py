""" Dataset generation Example usage """

from document import BaseDBLoader
from qa_generation import generate_qa 
import os
import json

import time
from datetime import datetime

# ## corpus 생성까지
# loader = BaseDBLoader()
# document = loader.load(is_regex=True, is_split=False)
# # print(document)
# corpus = loader.get_corpus()
# with open('result.json', 'w', encoding='utf-8') as file :
#     json.dump(corpus, file, indent='\t', ensure_ascii=False)
# print(len(corpus))
# # 464


### Dataset generation main
directory = os.path.dirname(__file__)
os.chdir(directory)

##### 데이터셋 10개씩 batch 나눠서 진행하였음.
folder_path = "./batch"
folder_result = "./resultQA"

if not os.path.exists(folder_result) :
    os.makedirs(folder_result)

for number, filename in enumerate(os.listdir(folder_path), start=26):
    #make result folder first
    print("="*60)
    print (f"dataset generate on batch_{number} start on : {datetime.now()}\n")

    if filename.endswith('.json'):
        file_path = os.path.join(folder_path, filename)

        with open(file_path, 'r', encoding="utf-8") as file:
            data = json.load(file)
        with open(f"./resultQA/batch_{number}_result.json", 'w+', encoding="utf-8") as result_file:
            json.dump(generate_qa(data, num_questions_per_chunk=25), result_file, indent="\t", ensure_ascii=False)
    time.sleep(3)
                

# with open('./langchain/result.json','r', encoding='utf-8') as file :  
#     loader = json.load(file)

# with open("qa_dataset.json", "w+", encoding="utf-8") as file:
#     json.dump(generate_qa(loader, num_questions_per_chunk=20), file, indent='\t', ensure_ascii=False)
    

