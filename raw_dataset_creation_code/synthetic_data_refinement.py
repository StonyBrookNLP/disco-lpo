"""
Synthetic Data Refinement using Static Analysis Feedback

This script refines existing synthetic code datasets by:
1. Running static analysis tools (Bandit, CodeQL) on secure code
2. Identifying remaining security issues
3. Using GPT-4 to generate more secure versions based on analysis feedback

Usage:
    python synthetic_data_refinement.py <input_csv> <output_csv>

Example:
    python synthetic_data_refinement.py initial_dataset.csv refined_dataset.csv

Input CSV Requirements:
    - 'Secure Code': Column containing secure code snippets
    - 'Bandit Feedback': Column containing Bandit analysis results  
    - 'Codeql Feedback': Column containing CodeQL analysis results

Output:
    - Enhanced dataset with 'More Secure Code' and reasoning columns
"""

import numpy as np
import pandas as pd
import random
import backoff
from openai import OpenAI
import openai
import logging
import argparse
import re

# Configure logging
logging.getLogger("urllib3").setLevel(logging.WARNING)

# Template for generating more secure code based on static analysis feedback
PROMPT_TEMPLATE = """The following is a Python code:
{}
The following are security issues identified by analyzer Bandit for this code:
{}
The following are security issues identified by analyzer CodeQL for this code:
{}
Using these rules and your knowledge about code security, generate the fixed code and a few sentences of reasoning.
Generate your answer in the following format:
FIX: <fixed code>
REASON: <reasoning on why this code is fixed>
"""


def argument_parsing():
    """
    Parse command line arguments for synthetic data refinement.
    
    Returns:
        argparse.Namespace: Parsed arguments containing:
            - input_filepath: Path to CSV with initial synthetic data
            - output_filepath: Path for refined output CSV
    """
    parser = argparse.ArgumentParser(
        description="Refine synthetic code dataset using static analysis feedback",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example usage:\n"
               "  python synthetic_data_refinement.py dataset.csv refined_dataset.csv"
    )
    parser.add_argument('input_filepath', type=str,
                       help='Input CSV file with synthetic data and analysis feedback')
    parser.add_argument('output_filepath', type=str,
                       help='Output CSV file for refined synthetic data')
    args = parser.parse_args()
    return args


def read_api_keys():
    """
    Read OpenAI API credentials from file.
    
    Returns:
        tuple: (api_key, organization_id)
        
    Note:
        Replace '#FILEPATH TO OPENAI API KEYS' with actual path to your credentials file.
    """
    try:
        with open("#FILEPATH TO OPENAI API KEYS", "r") as file:
            lines = file.read().splitlines()
            return lines[0], lines[1]
    except FileNotFoundError:
        raise FileNotFoundError("API key file not found. Please create the file with your OpenAI credentials.")



@backoff.on_exception(backoff.expo, openai.RateLimitError)
@backoff.on_exception(backoff.expo, openai.APIConnectionError)
def get_gpt4_output(input_prompt, client, model, temperature=0.001):
    """
    Generate response from OpenAI GPT-4 with retry logic.
    
    Args:
        input_prompt (str): The prompt to send to GPT-4
        client (OpenAI): Initialized OpenAI client
        model (str): Model name
        temperature (float): Sampling temperature
    
    Returns:
        str: Generated response from GPT-4
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": input_prompt}
        ],
        temperature=temperature
    )
    return response.choices[0].message.content


def prompt_creator(dataframe):
    """
    Create refinement prompts based on static analysis feedback.
    
    Args:
        dataframe (pd.DataFrame): DataFrame containing code and analysis feedback
        
    Returns:
        tuple: (prompts, markers) where:
            - prompts: List of refinement prompts for GPT-4
            - markers: Binary list indicating which codes need refinement
    """
    code = dataframe["Secure Code"].tolist()
    bandit = dataframe["Bandit Feedback"].tolist()
    codeql = dataframe["Codeql Feedback"].tolist()

    # Mark entries that have security issues identified
    markers = [0 for i in range(0, len(code))]

    # Check Bandit feedback
    for idx, item in enumerate(bandit):
        if item != "Nothing":
            markers[idx] = 1
    
    # Check CodeQL feedback  
    for idx, item in enumerate(codeql):
        if item != "Nothing":
            markers[idx] = 1

    # Create prompts only for codes with identified issues
    prompts = [
        PROMPT_TEMPLATE.format(code[i], bandit[i], codeql[i]) if markers[i] == 1 else "None" 
        for i in range(0, len(code))
    ]
    
    return prompts, markers



def remove_comments(code):
    """
    Remove comments from Python code.
    
    Args:
        code (str): Python code string
        
    Returns:
        str: Code with comments removed
    """
    # Remove single-line comments
    code = re.sub(r'#.*', '', code)
    
    # Remove multi-line comments (both '''...''' and """...""")
    code = re.sub(r'\'\'\'(.*?)\'\'\'', '', code, flags=re.DOTALL)
    code = re.sub(r'\"\"\"(.*?)\"\"\"', '', code, flags=re.DOTALL)
    
    return code

def extract_info_from_output(response_string):
    """
    Extract refined code and reasoning from GPT-4 response.
    
    Args:
        response_string (str): GPT-4 response containing fixed code and reasoning
        
    Returns:
        tuple: (new_code, reasoning) extracted from the response
    """
    # Find code block boundaries
    new_code_starts = response_string.find("```python", 0) + len("```python")
    new_code_ends = response_string.find("```", new_code_starts)
    
    # Extract and clean code
    new_code = response_string[new_code_starts:new_code_ends]
    new_code = remove_comments(new_code)
    new_code = new_code.strip()
    
    # Extract reasoning from remaining text
    reasoning = response_string[new_code_ends:].strip()
    
    return new_code, reasoning



if __name__ == "__main__":
    print("Starting synthetic data refinement...")
    args = argument_parsing()
    
    # Load input dataset
    print(f"Loading dataset from: {args.input_filepath}")
    dataset_df = pd.read_csv(args.input_filepath)
    print(f"Loaded {len(dataset_df)} entries")
    
    # Create refinement prompts
    print("Creating refinement prompts based on static analysis feedback...")
    prompts, markers = prompt_creator(dataset_df)

    # Initialize OpenAI client
    print("Initializing OpenAI client...")
    api_key, organization = read_api_keys()
    client = OpenAI(
        api_key=api_key,
        organization=organization,
    )

    # Display refinement statistics
    issues_found = sum(markers)
    print(f"Security issues found in {issues_found}/{len(markers)} code samples ({issues_found/len(markers)*100:.1f}%)")
    if issues_found > 0:
        print("Sample refinement prompt:")
        print(prompts[0][:200] + "...")

    # Generate refined code using GPT-4
    print("Generating refined code versions...")
    output = []
    model_name = "gpt-4o"
    
    for idx, prompt in enumerate(prompts):
        if idx % 10 == 0:
            print(f"Progress: {idx}/{len(prompts)} ({idx/len(prompts)*100:.1f}%)")
        
        if markers[idx] == 0:
            # No security issues found, no refinement needed
            output.append("Nothing")
        else:
            # Generate refined version using GPT-4
            try:
                refined_response = get_gpt4_output(prompt, client, model_name, 0.01)
                output.append(refined_response)
            except Exception as e:
                print(f"Error processing entry {idx}: {e}")
                output.append(f"ERROR: {str(e)}")

    # Process outputs and extract refined code and reasoning
    print("Processing generated responses...")
    refined_code = []
    refined_code_reasoning = []
    
    for idx, item in enumerate(output):
        if item == "Nothing":
            refined_code.append("Nothing")
            refined_code_reasoning.append("Nothing")
        elif item.startswith("ERROR:"):
            refined_code.append("ERROR")
            refined_code_reasoning.append(item)
        else:
            try:
                code, reasoning = extract_info_from_output(item)
                refined_code.append(code)
                refined_code_reasoning.append(reasoning)
            except Exception as e:
                print(f"Error parsing output for entry {idx}: {e}")
                refined_code.append("PARSE_ERROR")
                refined_code_reasoning.append(f"Parse error: {str(e)}")
    
    # Add refined columns to dataset
    dataset_df["More Secure Code"] = refined_code
    dataset_df["More Secure Code Reasoning"] = refined_code_reasoning
    
    # Save refined dataset
    dataset_df.to_csv(args.output_filepath, index=False)
    
    # Print completion statistics
    successful_refinements = sum(1 for code in refined_code if code not in ["Nothing", "ERROR", "PARSE_ERROR"])
    print(f"✓ Successfully refined {successful_refinements} code samples")
    print(f"✓ Results saved to: {args.output_filepath}")
    print("Synthetic data refinement completed!")


    