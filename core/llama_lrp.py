import torch
import json_lines
from lxt.models.llama import LlamaForCausalLM, attnlrp
from transformers import AutoTokenizer
import json
import matplotlib.pyplot as plt

# An even smaller LLaMA to use as an example
model_id = '/Users/tom/Downloads/mini-llama'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_length = 100000
prompt = "",
model = LlamaForCausalLM.from_pretrained(model_id, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
attnlrp.register(model)


def append_backward_hooks(m):
    rel_layer = {}

    def generate_hook(layer_name):
        def backward_hook(module, input_grad, output_grad):
            # cloning the relevance makes sure, it is not modified through memory-optimized LXT inplace operations if used
            rel_layer[layer_name] = output_grad[0].clone()

        return backward_hook

    # append hook to last activation of mlp layer
    for name, layer in m.named_modules():
        layer.register_full_backward_hook(generate_hook(name))
    return rel_layer


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


def read_source(path):
    sources = []
    with json_lines.open(path) as jsonl_file:
        for i, json_line in enumerate(jsonl_file):
            sources.append({
                "source_id": json_line["source_id"],
                "prompt": json_line["prompt"],
                "question": json_line["source_info"]["question"]
            })
    return sources


def check_and_zip(sources: list, responses: list):
    assert len(sources) == len(responses)
    sources.sort(key=lambda x: x["source_id"])
    responses.sort(key=lambda x: x["source_id"])
    result = []
    for (source, response) in zip(sources, responses):
        assert source["source_id"] == response["source_id"]
        result.append({
            "source_id": source["source_id"],
            "prompt": source["prompt"],
            "response": response["response"],
            "hallucination": response["hallucination"],
            "temperature": response["temperature"]
        })
    return result

def dump_json(data, file_name):
    with open(file_name, "w") as f:
        json.dump(data, f, indent=4)


def get_lxt_relevance(input_text, output_text, temperature=1, max_steps=500):
    input_ids = tokenizer.encode_plus(input_text, return_tensors="pt", add_special_tokens=True).input_ids
    input_ids_origin=input_ids.clone()
    input_ids.to(device)
    output_ids = tokenizer.encode_plus(output_text, return_tensors="pt", add_special_tokens=True).input_ids

    input_size, output_size = input_ids.shape[-1], output_ids.shape[-1]
    gen_length = 0
    relevances = []

    while gen_length < output_size and gen_length < max_steps:
        input_embeds = model.get_input_embeddings()(input_ids)
        output_logits = model(inputs_embeds=input_embeds.requires_grad_(), use_cache=False).logits
        # 引入温度参数
        next_token_logits = output_logits[0, -1, :] / temperature
        # 对logits应用softmax得到概率分布
        probs = torch.softmax(next_token_logits, dim=-1)
        # output token id
        # token_id = torch.multinomial(probs, num_samples=1).item()

        # check output token rank
        # output token
        # 获取排序后的索引
        sorted_indices = probs.argsort(descending=True)
        output_token_id = output_ids[:, gen_length]
        output_token = tokenizer.decode(output_token_id)
        rank = torch.where(sorted_indices == output_token_id)[0].item()
        print(f'{gen_length+1}th token, token={output_token}, rank={rank}', end="\n")

        # update input
        nxt_token_id = torch.tensor([[output_token_id]])
        nxt_token_id.to(device)
        input_ids = torch.cat((input_ids, nxt_token_id), dim=-1)

        # Get LXT relevance
        target_logits = next_token_logits[output_token_id]
        target_logits.backward(target_logits)
        relevance = input_embeds.grad.float().sum(-1).cpu()[0]

        # extend to same length
        relevance=relevance.cpu().tolist()
        relevance += [1] + [0] * (input_size + output_size - len(relevance) - 1)
        assert len(relevance) == input_size + output_size
        relevances.append(relevance)
        gen_length += 1
    return input_ids_origin.cpu().tolist(),output_ids.tolist(),relevances


if __name__ == '__main__':
    qa_source = read_source("../data/QA.jsonl")
    qa_hallucinated_response = read_response("../data/response_llama_7b_hallucination.jsonl", label=1)
    qa_nonhallucinated_response = read_response("../data/response_llama_7b_nonhallucination.jsonl", label=0)
    qa_pairs = check_and_zip(qa_source, qa_hallucinated_response + qa_nonhallucinated_response)
    for qa_pair in qa_pairs:
        input_ids,output_ids,qa_relevance=get_lxt_relevance(qa_pair["prompt"], qa_pair["response"],temperature=qa_pair["temperature"])
        dump_json({
            "prompt_ids":input_ids,
            "response_ids":output_ids,
            "relevance":qa_relevance,
            "source_id":qa_pair["source_id"],
            "prompt":qa_pair["prompt"],
            "response":qa_pair["response"],
            "temperature":qa_pair["temperature"],
            "hallucination":qa_pair["hallucination"]
        },f"./rag-{qa_pair['source_id']}-{qa_pair['hallucination']}.json")

