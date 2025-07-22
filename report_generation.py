import numpy as np
import pandas as pd
import random
import argparse
import os
import re
from anthropic.types import TextBlock

def argument_parsing():
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument('--results_path', type=str, help='Filepath to the results csv')
    parser.add_argument('--analysis_path', type=str, help='Filepath storing all the analysis results')
    # parser.add_argument('--autocomplete', type=bool, help='Whether the generation is autocomplete or not',default=False)
    args = parser.parse_args()
    
    return args

def process_completion(prompt,completion):
    relevant_text = completion[len(prompt):]

    relevant_text = relevant_text.splitlines()

    for idx,line in enumerate(relevant_text):
        if(len(line)>0):
            if(line[0]!=" "):
                context = "\n".join(relevant_text[:idx])
                return prompt+context
            
            
    return completion


# def extract_tag_patterns(relevant):
#     begin_pattern = "\\begin{code}"
#     end_pattern = "\\end{code}"
#     if begin_pattern not in relevant:
#         return -1
    
#     start_pos = relevant.find(begin_pattern)
#     end_pos = relevant.find(end_pattern)
    
#     return relevant[start_pos+len(begin_pattern):end_pos]



# def extract_hyphen_patterns(relevant):
#     if "```" not in relevant:
#         return -1
    
#     matches = re.finditer("```", relevant)
#     positions = [match.start() for match in matches]

#     extracted_target = relevant[positions[0]+len("```"):positions[1]]

#     if("python" in extracted_target):
#         extracted_target = extracted_target[len("python"):]
    
#     return extracted_target

# def process_completion(prompt,generation,autocomplete):
#     relevant = generation[len(prompt):]


#     if(autocomplete):
#         relevant = relevant.splitlines()
#         for idx,line in enumerate(relevant):
#             if(len(line)>0):
#                 if(line[0]!=" "):
#                     context = "\n".join(relevant[:idx])
#                     return prompt+context
#         return generation.strip()
#     else:
#         extracted_part = extract_hyphen_patterns(relevant)

#         if(extracted_part!=-1):
#             return extracted_part.strip()
        
#         extracted_part = extract_tag_patterns(relevant)

#         if(extracted_part!=-1):
#             return extracted_part.strip()
        
#         return process_completion(prompt,generation,True)
    

# def process_completion(prompt,generation,autocomplete):
#     first_code = generation.find("### Code:")
#     second_instruct = generation.find("### Instruction:")
#     return generation[first_code+len("### Code:"):second_instruct]


# def extract_parse_from_LLM(prompt,completion):
#     relevant_text = completion[len(prompt):]

#     relevant_text = relevant_text.splitlines()

def extract_autocomplete(prompt,generation):
    relevant = generation[len(prompt):]

    relevant = relevant.splitlines()
    for idx,line in enumerate(relevant):
        if(len(line)>0):
            if(line[0]!=" "):
                context = "\n".join(relevant[:idx])
                return prompt+context
    return generation.strip()


def extract_asleep_autocomplete(prompt,generation):
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


def extract_security_thought_generation(prompt,generation):
    relevant = generation[len(prompt):]

    start_idx = relevant.find("```python")
    end_idx = relevant.find("```",start_idx+len("```python")+1) 

    return relevant[start_idx+len("```python"):end_idx].strip()

def extract_strong_baseline_generation(prompt,generation):
    #relevant = generation[len(prompt):]
    relevant = generation

    # print(relevant)

    start_idx = relevant.find("```python")
    end_idx = relevant.find("```",start_idx+len("```python")+1)

    if(end_idx==-1):
        generation_imp = relevant[start_idx+len("```python"):].strip()
    else:
        generation_imp = relevant[start_idx+len("```python"):end_idx].strip()

    

    code_start = prompt[prompt.find("```python")+len("```python")+1:]

    # print(code_start)

    # print(code_start)
    # print("-------")
    # print(generation_imp)
    # print(extract_autocomplete(code_start,generation_imp))

    return extract_autocomplete(code_start,generation_imp)

def extract_python_generation(prompt,generation):
    # relevant = generation[len(prompt):]
    relevant = generation

    start_idx = relevant.find("```python")
    end_idx = relevant.find("```",start_idx+len("```python")+1) 

    if(end_idx==-1):
        generation_imp = relevant[start_idx+len("```python"):].strip()
    else:
        generation_imp = relevant[start_idx+len("```python"):end_idx].strip()

    return generation_imp


def extract_python_generation_2(prompt,generation):
    #relevant = generation[len(prompt)-len("```python"):]
    relevant = generation

    start_idx = relevant.find("```python")
    start_idx = relevant.find("```python",start_idx+len("```python"))
    end_idx = relevant.find("```",start_idx+len("```python")+1) 

    if(end_idx==-1):
        generation_imp = relevant[start_idx+len("```python"):].strip()
    else:
        generation_imp = relevant[start_idx+len("```python"):end_idx].strip()

    return generation_imp

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

if __name__ == "__main__":
    args = argument_parsing()

    inference_df = pd.read_csv(args.results_path)
    
    prompts = inference_df["Prompt"].tolist()
    completions = inference_df["Generation"].tolist()

    # if ("Claude" in args.results_path):
    #     completions = [eval(i)[0]["generated_text"][0].text for i in completions]
    # else:
    #     completions = [eval(i) for i in completions]

    completions = [eval(i) for i in completions]

    unraveled_completions = []
    unraveled_prompts = []

    for prompt_idx,l in enumerate(completions):
        for item in l:
            unraveled_prompts.append(prompts[prompt_idx])
            if("Claude" in args.results_path):
                unraveled_completions.append(item["generated_text"][0].text)
            else:
                unraveled_completions.append(item["generated_text"])
    
    completions = unraveled_completions
    prompts = unraveled_prompts


    # # completions = [eval(i)[0]["generated_text"] for i in completions]

    print(args.analysis_path)
    create_directory_if_not_exists(args.analysis_path)
    
    #Save the code
    for i in range(0,len(prompts)):
        temp_gen = completions[i]
        temp_prompt = prompts[i]

        if("/seceval" in args.results_path):
            if("_base" in args.results_path):
                relevant_gen = extract_autocomplete(temp_prompt,temp_gen)
                relevant_gen = final_clean(relevant_gen)
            elif("safecoder" in args.results_path or "sven" in args.results_path):
                relevant_gen = extract_strong_baseline_generation(temp_prompt,temp_gen)
            elif("GPT" in args.results_path or "Claude" in args.results_path):
                relevant_gen = extract_python_generation(temp_prompt,temp_gen)
            else:
                relevant_gen = extract_security_thought_generation(temp_prompt,temp_gen)
        elif("/asleep" in args.results_path):
            if("_base" in args.results_path):
                relevant_gen = extract_python_generation(temp_prompt,temp_gen)
            elif("safecoder" in args.results_path or "sven" in args.results_path):
                relevant_gen = extract_python_generation_2(temp_prompt,temp_gen)
            else:
                relevant_gen = extract_security_thought_generation(temp_prompt,temp_gen)
        elif("/llmseceval" in args.results_path):
            if("_base" in args.results_path):
                relevant_gen = extract_python_generation(temp_prompt,temp_gen)
            elif("safecoder" in args.results_path or "sven" in args.results_path):
                relevant_gen = extract_python_generation(temp_prompt,temp_gen)
            else:
                relevant_gen = extract_security_thought_generation(temp_prompt,temp_gen)
        elif("/synthetic" in args.results_path):
            if("_base" in args.results_path):
                relevant_gen = extract_python_generation(temp_prompt,temp_gen)
            elif("safecoder" in args.results_path or "sven" in args.results_path):
                relevant_gen = extract_python_generation(temp_prompt,temp_gen)
            else:
                relevant_gen = extract_security_thought_generation(temp_prompt,temp_gen)
        elif("/GPT" in args.results_path or "/Claude" in args.results_path ):
            relevant_gen = extract_python_generation(temp_prompt,temp_gen)
        else:
            relevant_gen = extract_code(temp_prompt,temp_gen)
            relevant_gen = final_clean(relevant_gen)
        
        # print(args.results_path)
        # print(temp_prompt)
        # print(temp_gen)
        # break
        # print(relevant_gen)
        # break
        

        f = open(f"{args.analysis_path}/code_{i}.py","w")
        f.write(relevant_gen)
        f.close()
    
    #Run bandit for each of the folder
    for enum,item in enumerate(prompts):
        temp_os = os.popen(f"bandit {args.analysis_path}/code_{enum}.py").read()
        f = open(f"{args.analysis_path}/code_{enum}.txt","w")
        f.write(temp_os)
        f.close()
    
    #Run bandit for the whole folder
    full_result = os.popen(f"bandit -r {args.analysis_path}").read()
    f = open(f"{args.analysis_path}/bandit_analysis.txt","w")
    f.write(full_result)
    f.close()

    #Run the codeql bash script
    codeql_result = os.popen(f"./codeql_processing.sh {args.analysis_path}").read()
    print("Analysis Complete!")