import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json_lines

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# 加载预训练的 LLaMA 模型和 tokenizer
model_name = "/root/hhc/llama-2-7b-chat"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
output_dir = "./prompt_llama"
source="./data/QA.jsonl"
response_hallucination="./data/response_llama_7b_hallucination.jsonl"
response_nonhallucination="./data/response_llama_7b_nonhallucination.jsonl"


class LlamaConfig:
    # 生成参数
    num_answers = 3  # 生成的答案数量
    max_length = 10000  # 生成文本的最大长度
    temperature = 1  # 温度参数，控制生成的随机性
    top_k = 50  # top-k 采样
    top_p = 0.95  # top-p (nucleus) 采样

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
def format_prompt(question, passages, answer):
    return f'Below is a question: {question} Below are related passages: {passages} Below is an answer: {answer} Your task is to determine whether the answer contains either or both of the following two types of hallucinations: 1. conflict: instances where the answer presents direct contraction or opposition to the passages; 2. baseless info: instances where the answer includes information which is not substantiated by or inferred from the passages. Then, compile the labeled hallucinated spans into a JSON dict, with a key "hallucination list" and its value is a list of hallucinated spans. If there exist potential hallucinations, the output should be in the following JSON format: {{"hallucination list": [hallucination span1, hallucination span2, ...]}}. Otherwise, leave the value as a empty list as following: {{"hallucination list": []}}. Output:'


def generate_answer(prompt):
    # 编码提示
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # 生成多个答案
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=LlamaConfig.max_length,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id
        )

    # 解码生成的答案
    answer = tokenizer.decode(output, skip_special_tokens=True)
    return answer


def main():
    global source,response_hallucination,response_nonhallucination
    source_info = read_source(source)
    hallucination_response = read_response(response_hallucination,label=0)
    nonhallucination_response = read_response(response_nonhallucination,label=1)
    responses = hallucination_response + nonhallucination_response
    responses.sort(key=lambda x: x["source_id"])
    for source, response in zip(source_info, responses):
        source_id = source["source_id"]
        question = source["question"]
        passages = source["passages"]
        answer = response["response"]
        prompt = format_prompt(question, passages, answer)
        llm_answer = generate_answer(prompt)
        with open(f"{output_dir}/rag-prompt-llama-{source_id}.txt", "w") as f:
            f.write(llm_answer)
            f.close()
        print(llm_answer)


if __name__ == '__main__':
    print(format_prompt("What is the capital of China?", "The capital of China is Beijing.", "Beijing"))
