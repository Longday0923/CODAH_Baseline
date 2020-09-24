import random
import pandas as pd
import numpy as np
import json
from tqdm import *


def split(full_list, shuffle=False, ratio=0.2):
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total == 0 or offset < 1:
        return [], full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:2 * offset]
    sublist_3 = full_list[2 * offset:]
    return sublist_1, sublist_2, sublist_3


def convert_to_codah_statement(qa_list: list, output_file1: str):
    print(f'converting qa_list to entailment dataset...')
    nrow = sum(1 for _ in qa_list)
    with open(output_file1, 'w') as output_handle1:
        # print("Writing to {} from {}".format(output_file, qa_file))
        for sample in tqdm(qa_list, total=nrow):
            output_dict = convert_sample_to_entailment(sample)
            output_handle1.write(json.dumps(output_dict))
            output_handle1.write("\n")
    print(f'converted statements saved to {output_file1}')
    print()


# Convert the QA file json to output dictionary containing premise and hypothesis
def convert_sample_to_entailment(sample: list):
    question_text = sample[1]
    choices = sample[2:6]  # left close right open
    single_qa_dict = {'class': sample[0], 'question': {'stem': sample[1]}, 'answer_key': sample[6]}
    choice_list = []
    choice_count = 0
    for choice in choices:
        statement = question_text + ' ' + choice
        create_output_dict(single_qa_dict, statement, choice_count == sample[6])
        choice_list.append({'text': choice, 'label': choice_count})
        choice_count += 1
    single_qa_dict['question']['choices'] = choice_list
    return single_qa_dict


# Create the output json dictionary from the input json, premise and hypothesis statement
def create_output_dict(input_json: dict, statement: str, label: bool) -> dict:
    if "statements" not in input_json:
        input_json["statements"] = []
    input_json["statements"].append({"label": label, "statement": statement})
    return input_json


if __name__ == "__main__":
    full_tsv = pd.read_csv('full_data.tsv', sep='\t')
    full_list = full_tsv.to_numpy()
    convert_to_codah_statement(full_list, 'codah.full.jsonl')
    # train, dev, test = split(full_list, shuffle=True, ratio=0.2)
    # convert_to_codah_statement(train, 'train.statement.jsonl')
    # convert_to_codah_statement(dev, 'train.statement.jsonl')
    # convert_to_codah_statement(test, 'train.statement.jsonl')
    print('Hey, there!')
