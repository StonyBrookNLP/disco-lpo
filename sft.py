import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments,BitsAndBytesConfig
from datasets import load_dataset
from trl import SFTTrainer
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model, prepare_model_for_kbit_training,PeftConfig,PeftModel
from utils import find_all_linear_names, print_trainable_parameters
import argparse
import re

#os.environ["WANDB_DISABLED"] = "true"


def argument_parsing():
    parser = argparse.ArgumentParser(description="Synthetic Code Generation")
    parser.add_argument('--train', type=str, help='Input csv file for the training data')
    parser.add_argument('--val', type=str, help='Input csv file for the validation data')
    parser.add_argument('--model', type=str, help='Huggingface model name')
    parser.add_argument('--adapted_model', type=str, default="", help='Adapter model')
    parser.add_argument('--adapter',action='store_true',help="Whether to use PEFT LORA adapters for training")
    parser.add_argument('--learning_rate', type=float, help='Learning rate for the SFT model')
    parser.add_argument('--epochs',type=int,help="Number of epochs for training the sft model")
    parser.add_argument('--out', type=str, help='Output directory for storing the model')
    parser.add_argument("--bnb",action='store_true',help='Whether 4 bit quantization is used')
    args = parser.parse_args()

    return args

if __name__=="__main__":

    args = argument_parsing()
    print(args)

    output_dir = args.out
    model_name = args.model

    train_dataset = load_dataset("csv",data_files=args.train,split="train")
    val_dataset = load_dataset("csv",data_files=args.val,split="train")

    if args.bnb :
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base_model = AutoModelForCausalLM.from_pretrained(model_name, device_map={"":"cuda"},torch_dtype="auto", quantization_config=bnb_config)
    else:
        base_model = AutoModelForCausalLM.from_pretrained(model_name,device_map={"":"cuda"},torch_dtype="auto")
    
    base_model.config.use_cache = False
    base_model = prepare_model_for_kbit_training(base_model)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = "right" 


    if(len(args.adapted_model)>0):
        config = PeftConfig.from_pretrained(args.adapted_model)
        peftmodel = PeftModel.from_pretrained(base_model, args.adapted_model)
        peftmodel = peftmodel.merge_and_unload()
        base_model = peftmodel
        print("Using adapted model for supervised finetuning!")

    # Setting up the LORA hyperparameters
    if(args.adapter):
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            # target_modules=[
            #         "q_proj",
            #         "v_proj",
            #         "k_proj",
            #         "out_proj",
            #         "fc_in",
            #         "fc_out",
            #         "wte",
            #     ],
            target_modules=list(set([name for name in re.findall(r'\((\w+)\): Linear', str(base_model.modules))])),
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )
        base_model = get_peft_model(base_model, peft_config)
    
    print_trainable_parameters(base_model)

    
    def formatting_prompts_func(example):
        #The format of the input should: instruction as a comment followed by the code
        output_texts = []

        for i in range(len(example['Secure Code'])):
            #text = f"### Instruction:\n{example['Instruction'][i]}\n### Answer:\n{example['Secure Code'][i]}"
            #text = f"'''\n{example['Instruction'][i]}\n'''\n{example['Secure Code'][i]}"
            #text = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{example['Instruction'][i]}\n\n### Response:\n```python\n{example['Secure Code'][i]}\n```"
            text = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{example['Instruction'][i]}\n\n### Security Thought:\n{example['Reasoning'][i]}\n\n### Response:\n```python\n{example['Secure Code'][i]}\n```"
            output_texts.append(text)

        
        print(output_texts[0])

        return output_texts

    # Parameters for training arguments details => https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py#L158
    training_args = TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        # gradient_checkpointing =True,
        max_grad_norm= 0.3,
        # max_steps = 100,
        num_train_epochs=args.epochs, 
        learning_rate=args.learning_rate,
        eval_strategy = "steps",
        eval_steps = 100,
        # bf16=True,
        save_total_limit=3,
        logging_steps=20,
        output_dir=output_dir,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05
    )

    trainer = SFTTrainer(
        base_model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        max_seq_length=1024,
        formatting_func=formatting_prompts_func,
        args=training_args
    )

    trainer.train() 
    trainer.save_model(output_dir)

    output_dir = os.path.join(output_dir, "final_checkpoint")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)