import json_lines, json


def extract_task_from_source(source_jsonl='./source_info.jsonl', task_type="QA", output_jsonl='./QA.jsonl'):
    with json_lines.open(source_jsonl) as input_file, open(output_jsonl, 'w') as output_file:
        for i, json_line in enumerate(input_file):
            if json_line['task_type'] == task_type:
                output_file.write(json.dumps(json_line) + '\n')


# extract_task_from_source()

def get_source_ids(source_jsonl='./QA.jsonl'):
    source_ids = []
    with json_lines.open(source_jsonl) as input_file:
        for i, json_line in enumerate(input_file):
            source_ids.append(json_line['source_id'])
    return source_ids


def extract_model_response(source_ids,
                           response_jsonl='./response.jsonl',
                           model_names=['mistral-7b-instruct', ],
                           hallucination=True,
                           output_jsonl='./mistral-7B-instruct_hallucination.jsonl'):
    with json_lines.open(response_jsonl) as input_file, open(output_jsonl, 'w') as output_file:
        for i, json_line in enumerate(input_file):
            if json_line['source_id'] in source_ids and json_line['model'].lower() in model_names:
                if hallucination and len(json_line['labels']) > 0:
                    output_file.write(json.dumps(json_line) + '\n')
                if not hallucination and len(json_line['labels']) == 0:
                    output_file.write(json.dumps(json_line) + '\n')


ids = get_source_ids()
extract_model_response(ids)
