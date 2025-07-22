# 0. imports
import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from accelerate import Accelerator
from datasets import Dataset, load_dataset
from peft import (
    LoraConfig,
    PeftType,
    PromptEncoderConfig,
    PeftConfig,
    PeftModel,
    PeftModelForFeatureExtraction,
    PeftModelForSequenceClassification
)
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments, set_seed

from modules.cpo_trainer import CPOTrainer
from modules.custom_cpo_trainer import LocalizedCPOTrainer
from trl import CPOConfig

# from trl import DPOTrainer
# from custom_dpo_trainer import CustomDPOTrainer
# from spo_trainer import SPOTrainer
# from custom_spo_trainer import CustomSPOTrainer
# from sft_utils import print_trainable_parameters

import pandas as pd
import re


# avoid = [1614,1923,2795,2988,5191,6097,6322,6846,8018]

# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    # data parameters
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for preference optimization"})
    gamma: Optional[float] = field(default=1.0, metadata={"help": "the gamma parameter for preference optimization"})
    loss_mask_val: Optional[float] = field(default=0.9, metadata={"help": "the value of importance on the relevant statements in the preference loss mask"})
    loss_type: Optional[str] = field(default="cpo", metadata={"help": "the type of loss function to use"})
    instruct: Optional[bool] = field(default=False, metadata={"help": "whether the model used is instruct or not"})

    # training parameters
    peft_model_path: Optional[str] = field(
        default="models/codellama-sft",
        metadata={"help": "the location of the SFT model name or path"},
    )

    base_model_path: Optional[str] = field(
        default="codellama/CodeLlama-7b-Instruct-hf",
        metadata={"help": "the huggingface directory of the base model name"},
    )

    learning_rate: Optional[float] = field(default=5e-5, metadata={"help": "optimizer learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    warmup_steps: Optional[int] = field(default=10, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})

    per_device_train_batch_size: Optional[int] = field(default=1, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "eval batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(
        default=16, metadata={"help": "the number of gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )

    gradient_checkpointing_use_reentrant: Optional[bool] = field(
        default=False, metadata={"help": "whether to use reentrant for gradient checkpointing"}
    )

    lora_alpha: Optional[float] = field(default=32, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.1, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=16, metadata={"help": "the lora r parameter"})

    max_prompt_length: Optional[int] = field(default=256, metadata={"help": "the maximum prompt length"})
    max_length: Optional[int] = field(default=1024, metadata={"help": "the maximum sequence length"})
    #max_steps: Optional[int] = field(default=2000, metadata={"help": "max number of training steps"})
    epochs: Optional[int] = field(default=1, metadata={"help": "max number of training steps"})
    logging_steps: Optional[int] = field(default=10, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=50, metadata={"help": "the saving frequency"})
    eval_steps: Optional[int] = field(default=50, metadata={"help": "the evaluation frequency"})

    output_dir: Optional[str] = field(default="./masked_dpo", metadata={"help": "the output directory"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "whether to load the model in 4bit"})
    model_dtype: Optional[str] = field(
        default="bfloat16", metadata={"help": "model_dtype[float16, bfloat16, float] for loading."}
    )

    # instrumentation
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "only train on 1000 samples"})
    report_to: Optional[str] = field(
        default="wandb",
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )
    seed: Optional[int] = field(
        default=42, metadata={"help": "Random seed that will be set at the beginning of training."}
    )

    train_path: Optional[str] = field(default="./datasets/synth_train", metadata={"help": "the training csv filepath"})
    eval_path: Optional[str] = field(default="./datasets/synth_val", metadata={"help": "the evaluation csv filepath"})
    use_masked_po: Optional[bool] = field(default=True, metadata={"help": "Whether to use regular dpo or masked dpo"})
    load_peft_model: Optional[bool] = field(default=False, metadata={"help": "Whether to use sft model as directed in the path"})




def get_prompts(str1,str2):
    lines1 = str1.splitlines()
    lines2 = str2.splitlines()

    common = []

    minlength = min(len(lines1),len(lines2))

    for i in range(0,minlength):
        if(lines1[i]==lines2[i]):
            common.append(lines1[i])
        else:
            break

    str1_uncommon = lines1[i:]
    str2_uncommon = lines2[i:]

    common = "\n".join(common)
    str1_uncommon = "\n".join(str1_uncommon)
    str2_uncommon = "\n".join(str2_uncommon)

    return common,str1_uncommon,str2_uncommon

def get_imp_lines(sec_code,vul_code):
    sec_code_lines = set(sec_code.splitlines())
    vul_code_lines = set(vul_code.splitlines())

    intersection = vul_code_lines & sec_code_lines

    sec_code_imp = list(sec_code_lines-intersection)
    vul_code_imp= list(vul_code_lines-intersection)

    if len(sec_code_imp) == 0 :
        sec_code_imp = [""]
    
    if len(vul_code_imp) == 0 :
        vul_code_imp = [""]
    
    return sec_code_imp,vul_code_imp



def get_dataset(filepath,args) -> Dataset:
    """Load the stack-exchange-paired dataset from Hugging Face and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }

    Prompts are structured as follows:
      "Question: " + <prompt> + "\n\nAnswer: "
    """
    dataset = load_dataset("csv",data_files=filepath,split="train")

    # if "train" in filepath:
    #     indices_to_keep = [i for i in range(0,len(dataset)) if i not in avoid ]
    #     dataset = dataset.select(indices_to_keep)
    original_columns = dataset.column_names



    def return_prompt_and_responses(samples) -> Dict[str, str]:
        secure_code_list = samples["Secure Code"]
        vulnerable_code_list = samples["Vulnerable Code"]
        instruction_list = samples["Instruction"]
        reasoning_list = samples["Reasoning"]

        #cases = [get_prompts(secure_code_list[i],vulnerable_code_list[i]) for i in range(0,len(samples["Secure Code"]))]
        imps = [get_imp_lines(secure_code_list[i],vulnerable_code_list[i]) for i in range(0,len(samples["Secure Code"]))]

        chosen = []
        rejected = []
        prompts = []
        chosen_imp = []
        rejected_imp = []

 
        # for i in cases:
        #     prompts.append(i[0])
        #     chosen.append(i[1])
        #     rejected.append(i[2])

        # for i in range(0,len(secure_code_list)):
        #     prompts.append(f"'''\n{instruction_list[i]}\n'''")
        #     chosen.append(f"\n{secure_code_list[i]}")
        #     rejected.append(f"\n{vulnerable_code_list[i]}")

        # for i in range(0,len(secure_code_list)):
        #     prompts.append(f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction_list[i]}### Response:\n")
        #     chosen.append(f"```python{secure_code_list[i]}```")
        #     rejected.append(f"```python{vulnerable_code_list[i]}```")

        for i in range(0,len(secure_code_list)):
            prompts.append(f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction_list[i]}\n\n### Security Thought:")
            chosen.append(f"\n{reasoning_list[i]}\n\n### Response:\n```python\n{secure_code_list[i]}\n```")
            rejected.append(f"\n{reasoning_list[i]}\n\n### Response:\n```python\n{vulnerable_code_list[i]}\n```")
        
        for i in imps:
            chosen_imp.append(i[0])
            rejected_imp.append(i[1])

        return {
            "prompt": prompts,
            "chosen": chosen,
            "rejected": rejected,
            "chosen_imp":chosen_imp,
            "rejected_imp":rejected_imp
        }

    return dataset.map(
        return_prompt_and_responses,
        batched=True,
        remove_columns=original_columns,
        load_from_cache_file=False
    )


if __name__ == "__main__":
    device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    set_seed(script_args.seed)

    

    # 1. load a pretrained model
    # torch_dtype = torch.float
    # if script_args.model_dtype == "float16":
    #     torch_dtype = torch.float16
    # elif script_args.model_dtype == "bfloat16":
    #     torch_dtype = torch.bfloat16
    torch_dtype = "auto"

    if(script_args.load_in_4bit):
        model = AutoModelForCausalLM.from_pretrained(
            script_args.base_model_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch_dtype,
            load_in_4bit=script_args.load_in_4bit,
            device_map={"":"cuda"}
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            script_args.base_model_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch_dtype,
            device_map={"":"cuda"}
        )

    model.config.use_cache = True

    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    tokenizer = AutoTokenizer.from_pretrained(script_args.base_model_path)
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Load the LoRA model if it is called for in arguments
    if(script_args.load_peft_model):
        config = PeftConfig.from_pretrained(script_args.peft_model_path)
        peftmodel = PeftModel.from_pretrained(model, script_args.peft_model_path)
        model = peftmodel.merge_and_unload()

    # 3. Load train and evaluation dataset
    train_dataset = get_dataset(script_args.train_path,script_args)
    eval_dataset = get_dataset(script_args.eval_path,script_args)

    print(len(train_dataset))
    print(train_dataset[0])
    print(len(eval_dataset))
    print(eval_dataset[0])
   


    # 4. initialize training arguments:
    training_args = CPOConfig(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        # max_steps=script_args.max_steps,
        num_train_epochs = script_args.epochs,
        logging_steps=script_args.logging_steps,
        save_steps=script_args.save_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        learning_rate=script_args.learning_rate,
        evaluation_strategy="steps",
        eval_steps=script_args.eval_steps,
        output_dir=script_args.output_dir,
        report_to=script_args.report_to,
        lr_scheduler_type=script_args.lr_scheduler_type,
        # warmup_steps=script_args.warmup_steps,
        warmup_ratio = 0.05,
        optim=script_args.optimizer_type,
        bf16=True,
        remove_unused_columns=True,
        # run_name="po",
        gradient_checkpointing_kwargs=dict(use_reentrant=script_args.gradient_checkpointing_use_reentrant),
        seed=script_args.seed,
        save_total_limit=3
    )

    peft_config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        # target_modules=[
        #     "q_proj",
        #     "v_proj",
        #     "k_proj",
        #     "out_proj",
        #     "fc_in",
        #     "fc_out",
        #     "wte",
        # ],
        target_modules=list(set([name for name in re.findall(r'\((\w+)\): Linear', str(model.modules))])),
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 5. initialize the PO trainer
    if(not script_args.use_masked_po):
        dpo_trainer = CPOTrainer(
            model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            peft_config=peft_config,
            # max_length=script_args.max_length,
            loss_type=script_args.loss_type,
            beta=script_args.beta,
            gamma=script_args.gamma
        )
    else:
        dpo_trainer = LocalizedCPOTrainer(
            model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            peft_config=peft_config,
            # max_length=script_args.max_length,
            loss_type=script_args.loss_type,
            beta=script_args.beta,
            gamma=script_args.gamma,
            loss_mask_val=script_args.loss_mask_val
        )

    # 6. train
    print(script_args)
    dpo_trainer.train()
    dpo_trainer.save_model(script_args.output_dir)

    # 7. save
    output_dir = os.path.join(script_args.output_dir, "final_checkpoint")
    dpo_trainer.model.save_pretrained(output_dir)
