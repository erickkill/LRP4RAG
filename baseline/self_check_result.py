response_hallucination = "../data/response_llama_13b_hallucination.jsonl"
response_nonhallucination = "../data/response_llama_13b_nonhallucination.jsonl"
# ['gpt','llama']
model = "gpt"
baseline_root = f"../baseline_output/llama_13b/selfcheck_{model}_out/"

import os
import json_lines
import json


def has_hallucination(path):
    threshold = 0.2
    with open(path, 'r') as f:
        sent_scores_prompt = json.load(f)["sent_scores_prompt"]
    for sent_score in sent_scores_prompt:
        if sent_score >= threshold:
            return True
    return False


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
    if not os.path.exists(f'{baseline_root}/{source_id}.json'):
        labels.append(label)
        preds.append(label)
        continue
    pred = has_hallucination(f'{baseline_root}/{source_id}.json')
    labels.append(label)
    preds.append(pred)

from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score

print(accuracy_score(labels, preds))
print(precision_score(labels, preds))
print(recall_score(labels, preds))
print(f1_score(labels, preds))
