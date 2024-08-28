import os.path

response_hallucination = "../data/response_llama_13b_hallucination.jsonl"
response_nonhallucination = "../data/response_llama_13b_nonhallucination.jsonl"
# ['gpt','llama']
model = "llama"
baseline_root = f"../baseline_output/llama_13b/prompt_{model}"

import json_lines


def has_hallucination(path):
    with open(path, "r") as f:
        text = f.read()
    if "\"hallucination list\": " not in text:
        return 0
    if "\"hallucination list\": []" in text or "\"hallucination list\": [...]" in text:
        return 0
    if len(text) <= 50:
        return 0
    return 1


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


responses = read_response(response_nonhallucination, label=0) + read_response(response_hallucination, label=1)
labels, preds = [], []
for response in responses:
    source_id = response["source_id"]
    label = response["hallucination"]
    if not os.path.exists(f'{baseline_root}/rag-prompt-{model}-{source_id}.txt'):
        pred = 0
    else:
        pred = has_hallucination(f'{baseline_root}/rag-prompt-{model}-{source_id}.txt')
    labels.append(label)
    preds.append(pred)

from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score

print(accuracy_score(labels, preds))
print(precision_score(labels, preds))
print(recall_score(labels, preds))
print(f1_score(labels, preds))
