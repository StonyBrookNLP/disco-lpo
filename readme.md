# Teaching an Old LLM Secure Coding: Localized Preference Optimization on Distilled Preferences

This is the Github Repository for the paper "Teaching an Old LLM Secure Coding: Localized Preference Optimization on Distilled Preferences" [(Link)](https://arxiv.org/abs/2506.00419) that has been accepted to **ACL 2025**.

## Abstract

LLM generated code often contains security
issues. We address two key challenges in improving secure code generation. First, obtaining high quality training data covering a broad
set of security issues is critical. To address this,
we introduce a method for distilling a preference dataset of insecure and secure code pairs
from frontier LLMs, along with a security reasoning that explains the issues and the fix. The
key idea here is to make use of security knowledge sources to devise a systematic prompting
strategy that ensures broad coverage. Second,
aligning models to secure code requires focusing on localized regions of code. Direct preference optimization methods, like SimPO, are
not designed to handle these localized differences and turn out to be ineffective. We address
this with a new localized preference optimization algorithm that masks the security related
tokens in both the winning (secure) and losing (insecure) responses. To prevent loss in
code quality, we also add a regularizer. Evaluations show that both training on our dataset,
DiSCo, and the new preference optimization
algorithm, LPO, yield substantial reductions in
code insecurity while also improving overall
code quality. 

![](./img/disco-lpo.png)

## Installation
Python Version : 3.10.14

The required libraries are in requirements.txt

``` 
pip install -r requirements.txt
```

## Data

Evaluation datasets are available in the ./eval folder.
DiSCo generated datasets are available in Huggingface at the following link: 
https://huggingface.co/datasets/StonyBrookNLP/DiSCo

## Models
Starcoder (best model) adapter modules are available in Huggingface at the following link:

SFT on DiSCo: https://huggingface.co/StonyBrookNLP/StarCoder2-SFT

LPO on DiSCo: https://huggingface.co/StonyBrookNLP/StarCoder2-LPO

## Training

### Supervised Fine-Tuning

Use sft.py in order to train a model on a dataset using supervised fine-tuning. Here is a sample command:

```
python supervised_fine_tuning.py --train datasets/DiSCo_train.csv --val datasets/DiSCo_val.csv --model bigcode/starcoder2-7b --adapter --out models/starcoder2-sft --bnb --learning_rate 1e-4 --epochs 2
```

### Localized Preference Optimization

Use pref_op.py in order to train a model on a dataset using localized preference optimization. Here is a sample command:

```
python pref_op.py --base_model_path bigcode/starcoder2-7b --peft_model_path models/starcoder2-sft --train_path datasets/synth_train.csv  --eval_path datasets/synth_val.csv  --loss_type simpo-kl --beta 10.0 --loss_mask_val  0.999999 --learning_rate 1e-5 --gamma 5.4 --use_masked_po True --load_peft_model True --output_dir models/starcoder2-lpo
```

### Merging with Base Model
In order to use the model for downstream generation, it is best to merge the adapters with the base model. This can be done using the "merge_peft_model.py" script. Place the appropriate values inside it and execute it to get your merged model.

P.S. To use LPO adapter for downstream generation, you must use the sft model merged with the original model as the base model for the adapter.

## Evaluation
Evaluation pipeline consists of two parts: code generation & metric calculation. 

Code generation involves using the LLMs to generate code given the prompts in the files present in "./eval/"

Metric calculation involves doing security analysis and getting the security report or calculating the code generation pass@k.

### Generation
Use inference.py to generate the code results for each evaluation dataset in "./eval/". Here is a code example:

```
python inference.py --base_model models/starcoder2-sft-merged --adapter True --peft_model models/starcoder2-lpo --test_path datasets/security_eval.csv --output_path results/starcoder2_lpo.csv --parses 5 --T 0.4 --max_new_tokens 512 --batch_size 4
```

### Testing

#### Security
If you are testing for security, then install bandit and download and unzip the codeql repository in this [link](https://drive.google.com/file/d/1uOayX1uYa2hE7CBtBfDQocXWMDuAAsOr/view?usp=sharing). Also allow codeql_processing.sh to be executable. Then run a command using report_generation.py as follows:

```
python report_generation.py --results_path results/starcoder2_lpocsv --analysis_path results/sec_gen_reports/starcoder2_lpo/
```

Then use security_metric.ipynb to get calculate and get the metric from the reports.

#### Code Generation
To measure the pass@k for code generation evaluation datasets, use coding_eval_analysis.py in the following manner to generate report:

```
python coding_eval_analysis.py --results_path results/starcoder2_lpo.csv --analysis_path results/code_gen_reports/starcoder2_lpo/
```

Afterwards, use code_gen_metric.ipynb to calculate the metrics from the report.

## Generating your own data
- The rules used to create the synthetic data are present in "./rules"
- However, due to the sensitive nature of the generated vulnerable code and the security risks that it poses, we will not be releasing the synthetic data generation code publicly. However, if you do wish to generate your own synthetic data for academic purposes, please reach out to "mdshasan@cs.stonybrook.edu" and we will provide you with synthetic data generation codebase. Thanks for understanding.

## Citation
Please include the following citation if you are using resources provided in this work:

```
@article{saqib2025teaching,
  title={Teaching an Old LLM Secure Coding: Localized Preference Optimization on Distilled Preferences},
  author={Saqib, Mohammad and Chakraborty, Saikat and Karmaker, Santu and Balasubramanian, Niranjan},
  journal={arXiv preprint arXiv:2506.00419},
  year={2025}
}
```

