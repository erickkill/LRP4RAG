import openai,json_lines,tqdm
from langchain.chat_models import ChatOpenAI
import os


source="./data/QA.jsonl"
response_hallucination="./data/response_llama_7b_hallucination.jsonl"
response_nonhallucination="./data/response_llama_7b_nonhallucination.jsonl"
api_base = "https://api.nextapi.fun/v1"
api_key = ""
output_dir = "../baseline_output/llama_7b/prompt_gpt"


def format_prompt(question, passages, answer):
    return f'Below is a question: {question} Below are related passages: {passages} Below is an answer: {answer} Your task is to determine whether the answer contains either or both of the following two types of hallucinations: 1. conflict: instances where the answer presents direct contraction or opposition to the passages; 2. baseless info: instances where the answer includes information which is not substantiated by or inferred from the passages. Then, compile the labeled hallucinated spans into a JSON dict, with a key "hallucination list" and its value is a list of hallucinated spans. If there exist potential hallucinations, the output should be in the following JSON format: {{"hallucination list": [hallucination span1, hallucination span2, ...]}}. Otherwise, leave the value as a empty list as following: \{{"hallucination list": []\}}. Output:'

def read_source(path):
    sources = []
    with json_lines.open(path) as jsonl_file:
        for i, json_line in enumerate(jsonl_file):
            sources.append({
                "source_id": json_line["source_id"],
                "prompt": json_line["prompt"],
                "question": json_line["source_info"]["question"],
                "passages": json_line["source_info"]["passages"]
            })
    return sources

def read_response(path, label=0):
    responses = []
    with json_lines.open(path) as jsonl_file:
        for i, json_line in enumerate(jsonl_file):
            responses.append({
                "hallucination": label,
                "response": json_line["response"],
                "source_id": json_line["source_id"],
                "temperature": json_line["temperature"]
            })
    return responses

def generate_answer(prompt):
    llm = ChatOpenAI(
        openai_api_base=api_base, # 注意，末尾要加 /v1
        openai_api_key=api_key,
    )
    res = llm.predict(prompt)
#    chat_completion = openai.chat.completions.create(
#        model="gpt-3.5-turbo", messages=[{"role": "user", "content":prompt}]
#    )

#    print(chat_completion.choices[0].message.content)
    print(res)
    return res



def main():
    global source,output_dir
    source_info = read_source(source)
    hallucination_response = read_response(response_hallucination,label=1)
    nonhallucination_response = read_response(response_nonhallucination,label=0)
    responses = hallucination_response + nonhallucination_response
    responses.sort(key=lambda x: x["source_id"])
    source_info.sort(key=lambda x: x["source_id"])
    for source, response in tqdm.tqdm(zip(source_info, responses)):
        assert source["source_id"]==response["source_id"]
        source_id = source["source_id"]
        question = source["question"]
        passages = source["passages"]
        answer = response["response"]
        prompt = format_prompt(question, passages, answer)
        llm_answer = generate_answer(prompt)
        with open(f"{output_dir}/rag-gpt-prompt-{source_id}.txt", "w") as f:
            f.write(llm_answer)
            f.close()
        print(llm_answer)


if __name__ == '__main__':
    main()
