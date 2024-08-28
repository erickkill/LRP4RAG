import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json_lines
import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 加载预训练的 LLaMA 模型和 tokenizer
model_name = "/root/hhc/llama-2-7b-chat"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True, torch_dtype=torch.bfloat16).to(device)
output_dir = "../baseline_output/llama_7b/llama_answer"



class LlamaConfig:
    # 生成参数
    num_answers = 5  # 生成的答案数量
    max_length = 10000  # 生成文本的最大长度
    temperature = 0.7  # 温度参数，控制生成的随机性
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
def gerenate_answers(prompt):
    # 编码提示
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_length = len(prompt)

    # 生成多个答案
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=LlamaConfig.max_length,
            num_return_sequences=LlamaConfig.num_answers,
            temperature=LlamaConfig.temperature,
            top_k=LlamaConfig.top_k,
            top_p=LlamaConfig.top_p,
            do_sample=True,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id
        )

    # 解码生成的答案
    answers = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    answers = [f'Answer{i + 1}: {x[prompt_length:]}' for i, x in enumerate(answers)]
    return answers


def main():
    for item in tqdm.tqdm(read_source("/root/hhc/LRP-eXplains-Transformers/data/QA.jsonl")):
        source_id, prompt = item['source_id'], item['prompt']
        fname = output_dir + f'/rag-llama-{source_id}.txt'
        answers = gerenate_answers(prompt=prompt)
        print(answers)
        with open(fname, 'w', encoding='utf-8') as f:
            f.write('\n'.join(answers))
            f.close()


if __name__ == "__main__":
    main()