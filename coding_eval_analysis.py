import numpy as np
import pandas as pd
import random
import argparse
import os
import subprocess
import re
from anthropic.types import TextBlock

import os
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# import ast

# def extract_first_function(code):
#     """
#     Extract the first function definition from Python code.
#     Returns the function's source code as a string.
#     """
#     try:
#         # Parse the code into an AST
#         tree = ast.parse(code)
        
#         # Find the first function definition
#         for node in ast.walk(tree):
#             if isinstance(node, ast.FunctionDef):
#                 # Get the line numbers
#                 start_line = node.lineno
#                 end_line = node.end_lineno
                
#                 # Split the original code into lines and get the function lines
#                 lines = code.split('\n')
#                 function_code = '\n'.join(lines[start_line-1:end_line])
#                 return function_code
                
#         return "No function found in the code"
    
#     except SyntaxError:
#         return "Invalid Python code"


# def extract_first_function(code):
#     relevant = code.splitlines()
#     first_line = relevant[0]
#     relevant = relevant[1:]
#     for idx,line in enumerate(relevant):
#         if(len(line)>0):
#             if(line[0]!=" "):
#                 context = "\n".join(relevant[:idx])
#                 return first_line+"\n"+context
#     return code.strip()


def argument_parsing():
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument('--results_path', type=str, help='Filepath to the results csv')
    parser.add_argument('--analysis_path', type=str, help='Filepath storing all the analysis results')
    # parser.add_argument('--autocomplete', type=bool, help='Autocomplete the following function',default=False)
    args = parser.parse_args()
    
    return args


def extract_autocomplete(prompt,generation):
    relevant = generation[len(prompt):]

    relevant = relevant.splitlines()
    for idx,line in enumerate(relevant):
        if(len(line)>0):
            if(line[0]!=" "):
                context = "\n".join(relevant[:idx])
                return prompt+context
    return generation.strip()


def extract_code(prompt,generation):
    relevant = generation[len(prompt):]
    
    endpoint = relevant.find("###")

    if(endpoint==-1):
        return generation
    else:
        return generation[:endpoint]

def final_clean(generation):
    startpoint = generation.find("### Code:")
    
    if startpoint==-1:
        return generation.strip()
    else:
        return generation[startpoint+len("### Code:"):].strip()

def execute_code(code_filepath):
    try:
        result = subprocess.run(['python',code_filepath], timeout=10)
        if(result.returncode!=0):
            return False
        else:
            return True
    except Exception as e:
        return False

# def extract_security_thought_generation(prompt,generation):
#     relevant = generation[len(prompt):]

#     start_idx = relevant.find("```python")
#     end_idx = relevant.find("```",start_idx+len("```python")+1) 

#     return relevant[start_idx+len("```python"):end_idx].strip()

def extract_secure_model_prompt_code(prompt,generation):
    # relevant = generation[len(prompt):]

    start_idx = generation.find("```python")
    end_idx = generation.find("```",start_idx+len("```python")+1) 

    return generation[start_idx+len("```python"):end_idx].strip()


def extract_secure_model_prompt_code_2(prompt,generation):
    # relevant = generation[len(prompt):]

    start_idx = generation.find("```python")
    # start_idx = generation.find("```python",start_idx+len("```python")+1)
    end_idx = generation.find("```",start_idx+len("```python")+1) 

    return generation[start_idx+len("```python"):end_idx].strip()



def extract_fs_model_prompt(prompt,generation):
    # relevant = generation[len(prompt):]

    start_idx = generation.find("```python")
    # start_idx = generation.find("```python",start_idx+len("```python")+1)
    end_idx = generation.find("```",start_idx+len("```python")+1) 

    return generation[start_idx+len("```python"):end_idx]

def extract_strong_baseline_generation(prompt,generation):
    #relevant = generation[len(prompt):]
    relevant = generation

    start_idx = relevant.find("```python")
    end_idx = relevant.find("```",start_idx+len("```python")+1)

    if(end_idx==-1):
        generation_imp = relevant[start_idx+len("```python"):].strip()
    else:
        generation_imp = relevant[start_idx+len("```python"):end_idx].strip()

    

    code_start = prompt[prompt.find("```python")+len("```python")+1:]

    # print(code_start)
    # print("-------")
    # print(generation_imp)
    # print(extract_autocomplete(code_start,generation_imp))

    return extract_autocomplete(code_start,generation_imp)

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


# def process_mbpp_tests(test_imports,test_list):
#     pattern = r"'([^']*)'"
#     tests = []
#     for i in range(0,len(test_list)):
#         temp_import = test_imports[i]
#         temp_test = test_list[i]

#         import_matches = re.findall(pattern, temp_import)
#         test_matches = re.findall(pattern,temp_test)

#         final_temp = "\n".join(import_matches)+"\n"+"\n".join(test_matches)
        
#         tests.append(final_temp)
#     return tests


def process_mbpp_tests(df_mbpp):
    test_imports = df_mbpp["test_imports"].tolist()
    test_list = df_mbpp["test_list"].tolist()
    test_imports = [eval(i) for i in test_imports]
    test_list = [eval(i) for i in test_list]

    final_test_imports = []
    for i in test_imports:
        temp = "\n"
        if(len(i)>0):
            for j in i:
                temp+=(j+"\n")
        final_test_imports.append(temp)

    final_test_list = []
    for i in test_list:
        temp = i[0].strip()
        temp = temp.split("assert")
        temp = [i.strip() for i in temp]
        x = ""
        for j in temp:
            if len(j)>0:
                x+="assert "+j+"\n"
        final_test_list.append(x)

    tests = []
    for i in range(0,len(test_list)):
        tests.append(final_test_imports[i]+final_test_list[i])
    
    return tests


def remove_main(x):
    #Write it as a state machine
    # lines = x.splitlines()

    # for idx,item in enumerate(lines):
    #     if "def" in item:
    #         return extract_autocomplete("\n".join(lines[:idx+1]),x)


    # return extract_first_function(x)
    idx1 = x.find("if __name__ ==")
    idx2 = x.find("def main")
    if idx1==-1 and idx2==-1:
        return x
    else:
        if(idx1==-1):
            idx=idx2
        elif(idx2==-1):
            idx=idx1
        else:
            idx = min(idx1,idx2)
        return x[:idx]
    
def remove_extra(x):
    #Write it as a state machine
    lines = x.splitlines()

    for idx,item in enumerate(lines):
        if "def" in item:
            return extract_autocomplete("\n".join(lines[:idx+1]),x)
        
def add_context(prompt,extracted_generation):
    context = prompt[prompt.find("```python")+len("```python"):]
    return context+"\n"+extracted_generation


def pass_at_k(n, c, k): 
    """ 
    Compute pass@k metric.
    
    :param n: total number of generated samples 
    :param c: number of correct solutions among generated samples
    :param k: the value of k in pass@k 
    :return: pass@k score
    """ 
    if n < k:  
        return 1.0 if c > 0 else 0.0  # If fewer than k solutions exist, return 1 if any are correct.

    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))  # Fixed missing parenthesis

    

if __name__ == "__main__":
    args = argument_parsing()

    inference_df = pd.read_csv(args.results_path)

    if("humaneval" in args.results_path):
        test_df = pd.read_csv("datasets/human_eval_x.csv")
        tests = test_df["Test"].tolist()
    
    if("mbpp" in args.results_path):
        test_df = pd.read_csv("datasets/mbpp.csv")
        # tests = process_mbpp_tests(test_df)
        tests = test_df["Test"].tolist()

    if("mbxp" in args.results_path):
        test_df = pd.read_csv("datasets/mbxp.csv")
        # tests = process_mbpp_tests(test_df)
        tests = test_df["Test"].tolist()
        base_incontext_examples = test_df["Base_Incontext_Examples"].tolist()[0]
        finetuned_incontext_examples =  test_df["Finetuned_Incontext_Examples"].tolist()[0]
    
    prompts = inference_df["Prompt"].tolist()
    completions = inference_df["Generation"].tolist()


    completions = [eval(i) for i in completions]

    unraveled_completions = []
    unraveled_prompts = []

    for prompt_idx,l in enumerate(completions):
        temp = []
        for item in l:
            # unraveled_prompts.append(prompts[prompt_idx])
            if("Claude" in args.results_path):
                temp.append(item["generated_text"][0].text)
            else:
                temp.append(item["generated_text"])
        unraveled_completions.append(temp)
    
    completions = unraveled_completions

    # if ("Claude" in args.results_path):
    #     completions = [eval(i)[0]["generated_text"][0].text for i in completions]
    # else:
    #     completions = [eval(i)[0]["generated_text"] for i in completions]

    prompt_compact = test_df["Prompt_Compact"].tolist()
   

    print(args.results_path)
    create_directory_if_not_exists(args.analysis_path)

    # prompts = prompts[:20]
    
    #Save the code
    exec_vals = []
    for i in range(0,len(prompts)):
        n = len(completions[0])
        c = 0
        for j in range(0,len(completions[i])):
            temp_gen = completions[i][j]
            temp_prompt = prompts[i]

            # if ("humaneval" in args.results_path):
            #     if("phi2" in args.results_path or "codellama_sft" in args.results_path or "mistral_sft" in args.results_path or "starcoder2_sft" in args.results_path):
            #         relevant_gen = extract_security_thought_generation(temp_prompt,temp_gen)
            #     else:
            #         relevant_gen = extract_autocomplete(temp_prompt,temp_gen)
            #         relevant_gen = final_clean(relevant_gen)
            # else:
            #     relevant_gen = extract_code(temp_prompt,temp_gen)
            #     relevant_gen = final_clean(relevant_gen)

            if("humaneval" in args.results_path):
                if("_base" in args.results_path):
                    relevant_gen = extract_autocomplete(temp_prompt,temp_gen)
                    relevant_gen = final_clean(relevant_gen)
                elif("safecoder" in args.results_path or "sven" in args.results_path):
                    relevant_gen = extract_strong_baseline_generation(temp_prompt,temp_gen)
                elif("GPT" in args.results_path or "Claude" in args.results_path):
                    relevant_gen = extract_fs_model_prompt(temp_prompt,temp_gen)
                    if("def" not in relevant_gen):
                        relevant_gen = add_context(temp_prompt,relevant_gen)
                else:
                    relevant_gen = extract_secure_model_prompt_code(temp_prompt,temp_gen)
                
                relevant_gen = remove_main(relevant_gen)
            elif("mbpp" in args.results_path):
                if("_base" in args.results_path):
                    relevant_gen = extract_secure_model_prompt_code(temp_prompt,temp_gen)
                else:
                    relevant_gen = extract_secure_model_prompt_code_2(temp_prompt,temp_gen)
                
                relevant_gen = remove_extra(relevant_gen)
            elif("mbxp" in args.results_path):
                if("_base" in args.results_path):
                    relevant_gen = extract_autocomplete(temp_prompt,temp_gen)
                    relevant_gen = final_clean(relevant_gen)
                    relevant_gen = relevant_gen.replace(base_incontext_examples,"")
                else:
                    temp_gen = temp_gen.replace(finetuned_incontext_examples,"")
                    relevant_gen = extract_secure_model_prompt_code(temp_prompt,temp_gen)
                
                relevant_gen = remove_main(relevant_gen)
            else:
                relevant_gen = extract_code(temp_prompt,temp_gen)
                relevant_gen = final_clean(relevant_gen)
            
            
            f = open(f"{args.analysis_path}/code_{i}_{j}.py","w")
            f.write(relevant_gen)
            # if("base" in args.results_path):
            f.write("\n\n")
            f.write(tests[i])
            f.close()

            if("mbxp" in args.results_path and i<3):
                continue
            res = execute_code(f"{args.analysis_path}/code_{i}_{j}.py")

            if(res):
                c+=1
        
        exec_vals.append([n,c])
    

    #Calculate pass@k values
    ks = [1,2,5] 
    pass_vals = [[],[],[]]

    print(len(exec_vals))
    print(exec_vals)

    for i in range(0,len(prompts)):
        for j in range(0,len(ks)):
            x=pass_at_k(exec_vals[i][0],exec_vals[i][1],ks[j])
            print(i,j,x)
            print(pass_vals)
            pass_vals[j].append(x)

    for j in range(0,len(ks)):
        pass_vals[j] = np.mean(pass_vals[j])
        # pass_vals[j]/len(prompts)
    
    print(pass_vals)

    final_file = open(f"{args.analysis_path}/pass_value.txt","w")
    
    for j in range(0,len(ks)):
        x = f"Pass@{ks[j]} (percentage):{pass_vals[j]}\n"
        print(x)
        final_file.write(x)
    
    final_file.close()

    
    
    # #Execute each of the code and paste the results in a text
    # pass_at_1 = 0
    # for enum,item in enumerate(prompts):
    #     if("mbxp" in args.results_path and enum<3):
    #         continue
    #     res = execute_code(f"{args.analysis_path}/code_{enum}.py")
    #     if(res):
    #         pass_at_1+=1
    
    # pass_at_1 = 100*pass_at_1/len(prompts)

    # print("Pass@1:",pass_at_1)

    # final_file = open(f"{args.analysis_path}/pass_value.txt","w")
    # final_file.write(f"Pass@1 (percentage):{pass_at_1:.2f}")
    # final_file.close()