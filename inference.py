import numpy as np
import pandas as pd
import random
from transformers import AutoTokenizer,AutoModelForCausalLM,AutoModel,set_seed
from transformers.pipelines.pt_utils import KeyDataset
from datasets import Dataset, load_dataset
import transformers
import torch
import os
from peft import (
    LoraConfig,
    PeftType,
    PromptEncoderConfig,
    PeftConfig,
    PeftModel,
    PeftModelForFeatureExtraction,
    PeftModelForSequenceClassification
)
import argparse
from tqdm import tqdm


set_seed(42)

def argument_parsing():
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument('--base_model', type=str, help='Trained model filepath')
    parser.add_argument('--peft_model', type=str, help='Trained model filepath')
    parser.add_argument('--test_path', type=str, help='Test dataset filepath')
    parser.add_argument('--adapter',type=bool,help="Whether to use PEFT LORA adapters for inference",default=False)
    parser.add_argument('--output_path', type=str, help='Generation output csv filepath')
    parser.add_argument("--bnb",type=bool,help='Whether 4 bit quantization is used',default=False)
    parser.add_argument("--parses",type=int,help='How many generations to parse',default=1)
    parser.add_argument("--T",type=float,help='Temperature of generation',default=0.2)
    parser.add_argument("--max_new_tokens",type=int,help='maximum new tokens to generate',default=512)
    parser.add_argument("--batch_size",type=int,help='batch size of the generations',default=8)
    args = parser.parse_args()

    return args

def create_directory_if_not_exists(filepath):
    # Extract the directory from the file path
    directory = os.path.dirname(filepath)
    
    # Check if the directory exists
    if not os.path.exists(directory):
        # Create the directory
        os.makedirs(directory)
        print(f"Directory '{directory}' created.")
    else:
        print(f"Directory '{directory}' already exists.")


def map_completion_as_instruction(x):
    x['Prompt'] = f"### Instruction:\nComplete the following Python function.\n### Code:\n{x['Prompt'].strip()}"
    return x



def map_as_instruction(x):
    x['Prompt'] = f"### Instruction:\n{x['Prompt'].strip()}\n### Response:\n```python"
    return x


def map_as_coding_instruction(x):
    x['Prompt'] = f"### Instruction:\n{x['Prompt'].strip()}\n### Response:\n```python\n{x['Prompt_Compact'].strip()}"
    return x


def map_as_secure_thinking_completion_instruction(x):
    x['Prompt'] = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nComplete the following Python code:\n```python\n{x['Prompt'].strip()}\n```\n\n### Security Thought:"
    return x

def map_as_alpaca_completion_instruction(x):
    x['Prompt'] = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nComplete the following Python code:\n```python\n{x['Prompt'].strip()}\n```\n\n### Response:"
    return x



def map_as_autocomplete(x):
    x['Prompt'] = f"{x['Prompt'].strip()}"
    return x

def map_as_autocomplete_with_examples(x):
    x['Prompt'] = f"{x['Base_Incontext_Examples']}\n\n{x['Prompt'].strip()}"
    return x
    


def map_as_python_autocomplete(x):
    x['Prompt'] = f"```python\n{x['Prompt'].strip()}"
    return x

def map_as_security_thinking(x):
    x['Prompt'] = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nWrite a Python function with declaration '{x['Func']}' to {x['Doc']}\n\n### Security Thought:"
    return x

def map_as_security_thinking_2(x):
    x['Prompt'] = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{x['Prompt'].strip()}\n\n### Security Thought:"
    return x

def map_as_coding_for_finetuned(x):
    x['Prompt'] = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{x['Doc']}\nThe function signature is '{x['Func']}'\n\n### Response:\n```python\n{x['Prompt_Compact']}"
    #x['Prompt'] = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nWrite a Python function with declaration'{x['Func']}' to {x['Doc']}\n\n### Response:\n```python\n{x['Prompt_Compact']}"
    #x['Prompt'] = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nWrite a Python function with declaration'{x['Func']}' to {x['Doc']}\n\n### Security Thought:"

    return x

def map_as_coding_for_finetuned_2(x):
    #x['Prompt'] = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{x['Doc']}\nThe function signature is '{x['Func']}'\n\n### Response:\n```python\ndef {x['Prompt_Compact']}"
    #x['Prompt'] = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{x['Prompt']}\n\n### Response:\n```python\n{x['Prompt_Compact'].strip()}"
    #x['Prompt'] = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{x['Prompt']}\n\n### Security Thought:"
    x['Prompt'] = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n{x['Finetuned_Incontext_Examples']}\n\n### Instruction:\n{x['Doc']}\nThe function signature is '{x['Func']}'\n\n### Response:\n```python\n{x['Prompt_Compact'].strip()}"
    return x


def format_prompt(test_path,output_path,test_data):
    if("security_eval" in test_path):
        if("_base" in output_path):
            test_data = test_data.map(map_as_autocomplete)
        else:
            test_data = test_data.map(map_as_security_thinking)

    if("asleep" in test_path):
        if("_base" in output_path):
            test_data = test_data.map(map_as_python_autocomplete)
        else:
            test_data = test_data.map(map_as_secure_thinking_completion_instruction)
    
    if("llm_seceval" in test_path):
        if("_base" in output_path):
            test_data = test_data.map(map_as_instruction)
        else:
            test_data = test_data.map(map_as_security_thinking_2)
    
    if("synth" in test_path):
        if("_base" in output_path):
            test_data = test_data.map(map_as_instruction)
        else:
            test_data = test_data.map(map_as_security_thinking_2)

    if("human_eval_x" in test_path):
        if("_base" in output_path):
            test_data = test_data.map(map_as_autocomplete)
        else:
            test_data = test_data.map(map_as_coding_for_finetuned)

    if("mbpp" in test_path):
        if("_base" in output_path):
            test_data = test_data.map(map_as_coding_instruction)
        else:
            test_data = test_data.map(map_as_coding_for_finetuned_2)
    
    if("mbxp" in test_path):
        # if("_base" in output_path):
        #     test_data = test_data.map(map_as_autocomplete_with_examples)
        # else:
        #     test_data = test_data.map(map_as_coding_for_finetuned_2)
        if("_base" in output_path):
            test_data = test_data.map(map_as_autocomplete)
        else:
            test_data = test_data.map(map_as_coding_for_finetuned)

    
    return test_data





if __name__ == "__main__":

    args = argument_parsing()
    print(args)

    #Create output directory
    create_directory_if_not_exists(args.output_path)

    #Load the model
    tokenizer = AutoTokenizer.from_pretrained(args.base_model,padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(args.base_model)
    if(args.adapter):
        peft_model_id = args.peft_model
        config = PeftConfig.from_pretrained(peft_model_id)
        inference_model = PeftModel.from_pretrained(model, peft_model_id)
        # inference_model = inference_model.merge_and_unload()
    else:
        inference_model = model

    print(inference_model)
    
    #Prepare the transformers pipeline
    pipeline = transformers.pipeline(
        "text-generation",
        model=inference_model,
        tokenizer=tokenizer,
        torch_dtype="auto",
        device="cuda",
        do_sample=True,
        top_k=50,
        temperature=args.T,
        top_p=0.95,
        # num_beams=args.parses,
        num_return_sequences=args.parses,
        # eos_token_id=tokenizer.eos_token_id,
        # pad_token_id=tokenizer.eos_token_id,
        max_new_tokens = args.max_new_tokens,
        # truncation = True
        use_cache = True
    )

    print(tokenizer.pad_token_id)
    print(tokenizer.eos_token_id)
    print(model.config.eos_token_id)


    pipeline.tokenizer.pad_token_id = pipeline.model.config.eos_token_id

    #Load and prep the dataset
    test_data = load_dataset("csv",data_files=args.test_path,split="train")

    # if(len(test_data)>20):
    #     test_data = test_data[:20]
        # test_data = test_data.shuffle(seed=42).select(range(200))
    
    # test_data = test_data.shuffle(seed=42).select(range(20))
    print(len(test_data))

    print(test_data[0])


    #Convert the prompts for each dataset

    # if ("security_eval" in args.test_path or "asleep_at_keyboard" in args.test_path or "human_eval_x" in args.test_path):
    # if("human_eval_x" in args.test_path or "security_eval" in args.test_path):
    #     if("phi2" in args.peft_model or "codellama-sft" in args.peft_model or "mistral-sft" in args.peft_model or "starcoder2-sft" in args.peft_model):
    #         if("human_eval_x" in args.test_path ):
    #             test_data = test_data.map(map_as_alpaca_completion_instruction)
    #         else:
    #             test_data = test_data.map(map_as_secure_thinking_completion_instruction)
    #     elif("orig_safecoder" in args.output_path):
    #         test_data = test_data.map(map_as_alpaca_completion_instruction)
    #     else:
    #         test_data = test_data.map(map_completion_as_automcomplete)

    # if("human_eval_x" in args.test_path):
    #     if("base" in args.test_path):
    #         test_data = test_data.map(map_as_secure_thinking_completion_instruction)
    #     else:
    #         test_data = test_data.map(map_as_alpaca_completion_instruction)
    
    # if("security_eval" in args.test_path):
    #     if("base" in args.test_path):
    #         test_data = test_data.map(map_completion_as_automcomplete)
    #     else:
    #         test_data = test_data.map(map_as_secure_thinking_completion_instruction)

    # if ("asleep_at_keyboard" in args.test_path):
    #     if(args.peft_model==None):
    #         test_data = test_data.map(map_completion_as_automcomplete)
    #     else:
    #         test_data = test_data.map(map_completion_as_instruction)
    
    # if("synthetic" in args.test_path or "mbpp" in args.test_path):
    #     test_data = test_data.map(map_as_instruction)

    test_data = format_prompt(args.test_path,args.output_path,test_data)

    prompts = test_data["Prompt"]
    # prompts = [item for item in prompts for _ in range(args.parses)]

    print(prompts[0])

    #Limiting for check
    # prompts = prompts[:20]
    # test_data = test_data[:args.batch_size*3]

    generations = []
    # i = 0
    for out in tqdm(pipeline(KeyDataset(test_data, "Prompt"),batch_size=args.batch_size),total=len(prompts)):
        temp_output = out
        generations.append(temp_output)
        # i+=1
        # if i >= 20:
        #     break
    
    print(generations)
    print(len(generations))
    #Save the generation as a csv
    df = pd.DataFrame({"Prompt":prompts[:len(generations)],"Generation":generations})
    df.to_csv(args.output_path)
    
    