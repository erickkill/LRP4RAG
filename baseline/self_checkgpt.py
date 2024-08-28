import json
import re

import torch
from selfcheckgpt.modeling_selfcheck import SelfCheckLLMPrompt
import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from selfcheckgpt.modeling_selfcheck import SelfCheckMQAG, SelfCheckBERTScore, SelfCheckNgram
from selfcheckgpt.modeling_selfcheck_apiprompt import SelfCheckAPIPrompt
import spacy
import json_lines
import os

nlp = spacy.load("en_core_web_sm")
api_base = "https://api.nextapi.fun/v1"
api_key = ""
os.environ["OPENAI_API_KEY"] = api_key
os.environ["OPENAI_BASE_URL"] = api_base

sampled_passages_path = "./llm_answer"
passage_hallucination_path = "./data/response_llama_7b_hallucination.jsonl"
passage_non_hallucination_path = "./data/response_llama_7b_nonhallucination.jsonl"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 判别模型
# model_name = "/root/hhc/llama-2-7b-chat"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name,local_files_only=True,torch_dtype=torch.bfloat16).to(device)

# selfcheck_prompt = SelfCheckLLMPrompt(model_name, device)
selfcheck_prompt = SelfCheckAPIPrompt(client_type="openai", model="gpt-3.5-turbo")
selfcheck_prompt.client.api_key = api_key
selfcheck_prompt.client.base_ur = api_base
selfcheck_mqag = SelfCheckMQAG(device=device)  # set device to 'cuda' if GPU is available
selfcheck_bertscore = SelfCheckBERTScore(rescale_with_baseline=True)
selfcheck_ngram = SelfCheckNgram(n=1)  # n=1 means Unigram, n=2 means Bigram, etc.
output = "../baseline_output/llama_7b/selfcheck_gpt_out"


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


def read_sample_passages(ppath, answer_num=5):
    answers = []
    with open(ppath, 'r') as f:
        text = str(f.read())
        for i in range(1, answer_num + 1):
            answeri = f'Answer{i}:'
            answer_nxt = f'Answer{i + 1}:'
            p, q = text.find(answeri), text.find(answer_nxt)
            if q > 0:
                answers.append(text[p + len(answeri):q].strip().replace("\n", ""))
            else:
                answers.append(text[p + len(answeri):].strip().replace("\n", ""))
        return answers


def selfcheck_prompt_predict(sentences, samples):
    sent_scores_prompt = selfcheck_prompt.predict(
        sentences=sentences,  # list of sentences
        sampled_passages=samples,  # list of sampled passages
        verbose=True,  # whether to show a progress bar
    )
    return sent_scores_prompt


def selfcheck_mqag_predict(passage, sentences, samples):
    sent_scores_mqag = selfcheck_mqag.predict(
        sentences=sentences,  # list of sentences
        passage=passage,  # passage (before sentence-split)
        sampled_passages=samples,  # list of sampled passages
        num_questions_per_sent=5,  # number of questions to be drawn
        scoring_method='bayes_with_alpha',  # options = 'counting', 'bayes', 'bayes_with_alpha'
        beta1=0.8, beta2=0.8,  # additional params depending on scoring_method
    )
    return sent_scores_mqag


def selfcheck_bertscore_predict(sentences, samples):
    sent_scores_bertscore = selfcheck_bertscore.predict(
        sentences=sentences,  # list of sentences
        sampled_passages=samples,  # list of sampled passages
    )
    return sent_scores_bertscore


def selfcheck_ngram_predict(passage, sentences, samples):
    sent_scores_ngram = selfcheck_ngram.predict(
        sentences=sentences,
        passage=passage,
        sampled_passages=samples,
    )
    return sent_scores_ngram


def main():
    responses = read_response(passage_hallucination_path, label=1) + read_response(passage_non_hallucination_path,
                                                                                   label=0)
    for response in tqdm.tqdm(responses):
        source_id = response["source_id"]
        if os.path.exists(f'{output}/{source_id}.json'):
            continue
        response_text = response["response"]
        sentences = [sent.text.strip() for sent in nlp(response_text).sents]
        samples = read_sample_passages(f'{sampled_passages_path}/rag-llama-{source_id}.txt')
        for i in range(3):
            try:
                sent_scores_prompt = selfcheck_prompt_predict(sentences, samples)
            except Exception as e:
                print(f'fail {i + 1} times')
            else:
                break

        #        sent_scores_mqag = selfcheck_mqag_predict(response_text, sentences, samples)
        #        sent_scores_bertscore = selfcheck_bertscore_predict(sentences, samples)
        #        sent_scores_ngram = selfcheck_ngram_predict(response_text, sentences, samples)
        with open(f'{output}/{source_id}.json', 'w') as f:
            json.dump({
                "sent_scores_prompt": sent_scores_prompt.tolist(),
                #                "sent_scores_mqag": sent_scores_mqag,
                #                "sent_scores_bertscore": sent_scores_bertscore,
                #                "sent_scores_ngram": sent_scores_ngram,
            }, f, indent=4)


if __name__ == '__main__':
    main()
