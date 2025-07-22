import os
import numpy as np
import pandas as pd
import random
from transformers import AutoTokenizer,AutoModelForCausalLM,AutoModel
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


#Load the model



base_model = "Path to your base model here"
peft_model = "Path to your adapted model here"
save_location = "Path to save the merged model here"


tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(base_model,torch_dtype="auto")

peft_model_id = peft_model
config = PeftConfig.from_pretrained(peft_model_id)
inference_model = PeftModel.from_pretrained(model, peft_model_id)
inference_model = inference_model.merge_and_unload()

inference_model.save_pretrained(save_location)
tokenizer.save_pretrained(save_location)

print(f"{save_location} done!")