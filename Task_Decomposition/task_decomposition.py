import torch
from torch.utils.data import DataLoader
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer
)


def Decomposition(args, task):
    # TODO: prompt, output_num
    # load prompt
    f = open("./Task_Decomposition/prompt_example.txt")
    lm_template = f.read()
    prompt = "\n".join([lm_template, f"TASK: {task}", "SUBGOALS: "])
    print(prompt)
    
    # load model from the hub
    tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-base')
    model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-base')

    inputs = tokenizer(prompt, return_tensors="pt").input_ids

    output_sequence = model.generate(input_ids=inputs, max_length=100)

    output = tokenizer.decode(output_sequence[0], skip_special_tokens=True)
    print(output)

    # sub-goal list
    temp = output.split('.')
    sub_goal_list = []
    for i in range(1, len(temp)):
        if i%2 == 0:
            continue
        else:
            sub_goal_list.append(temp[i][1:])
    return output, sub_goal_list
