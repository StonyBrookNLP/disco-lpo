"""
Synthetic Data Creation using OpenAI GPT-4

This script generates synthetic vulnerable and secure code pairs using OpenAI's GPT-4 API.
It takes prompts created by prompt_creation.py and generates structured code examples
with vulnerabilities and their corresponding fixes.

Usage:
    python synthetic_data_creation.py <input_csv> <output_csv> [--size N]

Example:
    python synthetic_data_creation.py prompts.csv results.csv --size 1000

Requirements:
    - OpenAI API key file at specified path
    - Input CSV with 'Prompt' column
    - Sufficient OpenAI API credits
"""

import numpy as np
import pandas as pd
import random
import backoff
from openai import OpenAI
import openai
import logging
import argparse
import json

# Configure logging to reduce noise from urllib3
logging.getLogger("urllib3").setLevel(logging.WARNING)
TEMP_FOLDER = "temp/"


def argument_parsing():
    """
    Parse command line arguments for synthetic data generation.
    
    Returns:
        argparse.Namespace: Parsed arguments containing:
            - input_filepath: Path to CSV file with prompts
            - output_filepath: Path for output CSV file
            - size: Number of prompts to process (-1 for all)
    """
    parser = argparse.ArgumentParser(
        description="Generate synthetic vulnerable/secure code pairs using GPT-4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example usage:\n"
               "  python synthetic_data_creation.py prompts.csv output.csv --size 100"
    )
    parser.add_argument('input_filepath', type=str, 
                       help='Input CSV file containing the prompts (must have "Prompt" column)')
    parser.add_argument('output_filepath', type=str, 
                       help='Output CSV file for generated code pairs')
    parser.add_argument('--size', type=int, 
                       help='Number of prompts to process (default: all prompts)', 
                       default=-1)
    args = parser.parse_args()
    return args


def read_api_keys():
    """
    Read OpenAI API credentials from file.
    
    Expected file format:
        Line 1: API key
        Line 2: Organization ID
    
    Returns:
        tuple: (api_key, organization_id)
    
    Note:
        Replace '#FILEPATH TO OPENAI API KEYS' with actual path to your API key file.
        Keep this file secure and never commit it to version control.
    """
    try:
        with open("#FILEPATH TO OPENAI API KEYS", "r") as file:
            lines = file.read().splitlines()
            return lines[0], lines[1]
    except FileNotFoundError:
        raise FileNotFoundError("API key file not found. Please create the file with your OpenAI credentials.")
    except IndexError:
        raise ValueError("API key file must contain at least 2 lines: API key and organization ID.")



@backoff.on_exception(backoff.expo, openai.RateLimitError)
@backoff.on_exception(backoff.expo, openai.APIConnectionError)
def get_gpt4_output(input_prompt, client, model, temperature=0.001):
    """
    Generate response from OpenAI GPT-4 with retry logic for rate limiting.
    
    Args:
        input_prompt (str): The prompt to send to GPT-4
        client (OpenAI): Initialized OpenAI client
        model (str): Model name (e.g., 'gpt-4o-2024-08-06')
        temperature (float): Sampling temperature for response generation
    
    Returns:
        str: Generated response from GPT-4
    
    Note:
        Uses exponential backoff for rate limit and connection errors.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": input_prompt}
            ],
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating response: {e}")
        raise




if __name__ == "__main__":
    print("Starting synthetic data generation...")
    args = argument_parsing()
    
    # Load input prompts
    print(f"Loading prompts from: {args.input_filepath}")
    try:
        prompts_df = pd.read_csv(args.input_filepath)
        print(f"Loaded {len(prompts_df)} prompts")
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file not found: {args.input_filepath}")

    # Select subset of prompts if specified
    if args.size == -1:
        prompts = prompts_df["Prompt"].tolist()
        print(f"Processing all {len(prompts)} prompts")
    else:
        if args.size > len(prompts_df):
            print(f"Warning: Requested size ({args.size}) exceeds available prompts ({len(prompts_df)}). Using all prompts.")
            prompts = prompts_df["Prompt"].tolist()
        else:
            sampled_df = prompts_df.sample(n=args.size, random_state=42)
            prompts = sampled_df["Prompt"].tolist()
        print(f"Processing {len(prompts)} prompts")

    # Initialize OpenAI client
    print("Initializing OpenAI client...")
    api_key, organization = read_api_keys()
    client = OpenAI(
        api_key=api_key,
        organization=organization,
    )

    # Generate synthetic code using GPT-4
    print("Generating synthetic code pairs...")
    output = []
    model_name = "gpt-4o-2024-08-06"
    
    for idx, prompt in enumerate(prompts):
        # Save progress every 10 iterations
        if idx % 10 == 0:
            print(f"Progress: {idx}/{len(prompts)} ({idx/len(prompts)*100:.1f}%)")
            if idx > 0:  # Don't save empty DataFrame on first iteration
                output_df = pd.DataFrame({"Prompt": prompts[:idx], "Output": output})
                output_df.to_csv(args.output_filepath, index=False)
                print(f"Temporary save completed: {args.output_filepath}")
        
        try:
            generated_response = get_gpt4_output(prompt, client, model_name, 0.01)
            output.append(generated_response)
        except Exception as e:
            print(f"Error processing prompt {idx}: {e}")
            output.append(f"ERROR: {str(e)}")
    
    # Final save
    print("Saving final results...")
    output_df = pd.DataFrame({"Prompt": prompts, "Output": output})
    output_df.to_csv(args.output_filepath, index=False)
    
    print(f"✓ Successfully generated {len(output)} code pairs")
    print(f"✓ Results saved to: {args.output_filepath}")
    print("Synthetic data generation completed!")
    
    # Print sample of last generated output for verification
    if output:
        print("\n--- Sample of last generated output ---")
        print(output[-1][:500] + "..." if len(output[-1]) > 500 else output[-1])


    