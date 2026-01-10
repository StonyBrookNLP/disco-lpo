"""
Prompt Creation for Synthetic Security Dataset Generation

This script creates prompts for generating synthetic code datasets containing
security vulnerabilities and their fixes. It combines security rules from 
various sources (Bandit, CodeQL, CWE) to create comprehensive prompts.

Author: Dataset Creation Team
Usage: python prompt_creation.py
Outputs: 
    - Simple prompts CSV file
    - Complex prompts CSV file
"""

import numpy as np
import pandas as pd
import random
from libraries import modules


# Helper functions for data extraction and processing
def extract_dataset(name):
    """
    Extract security rules dataset and add source information.
    
    Args:
        name (str): Name of the security tool (e.g., 'bandit', 'codeql', 'cwe')
    
    Returns:
        pd.DataFrame: DataFrame with security rules and source column
    """
    filepath = f"rules/{name}_rules.csv"
    df = pd.read_csv(filepath)
    df["From"] = [name for i in range(len(df))]
    return df

def get_relevant_stuff_from_rules(list_of_dfs):
    """
    Extract and combine relevant fields from multiple security rule datasets.
    
    Args:
        list_of_dfs (list): List of DataFrames containing security rules
    
    Returns:
        tuple: (comes_from, cwes, titles, descriptions) - extracted field lists
    """
    comes_from = []
    cwes = []
    titles = []
    descriptions = []
    for df in list_of_dfs:
        comes_from += df["From"].tolist()
        cwes += df["CWE"].tolist()
        titles += df["Title"].tolist()
        descriptions += df["Description"].tolist()
    return comes_from, cwes, titles, descriptions


if __name__ == "__main__":
    # Extract security rules from different sources
    # Note: Replace placeholder paths with actual file paths
    bandit_df = extract_dataset("#BANDIT_RULES_FILEPATH")
    codeql_df = extract_dataset("#CODEQL_RULES_FILEPATH")
    cwe_df = extract_dataset("#CWE_RULES_FILEPATH")

    # Extract relevant fields from all rule datasets
    comes_from, cwes, titles, descriptions = get_relevant_stuff_from_rules([bandit_df, codeql_df, cwe_df])

    # Template for basic prompt generation (legacy format)
    template = """The following is a security issue found in Python: '{}'. {}
    Generate a Python code containing this security issue and the fixed version of this code that makes the minimal change.
    The codes should be within 20-50 lines, in code blocks and the security issue should be as implicit 
    as possible.
    Then,generate the reasoning on how you fixed the code.
    Finally, generate a short (one or two sentence) instruction summary that can be used by a language model
    to generate either of the code. The instruction should not reveal the security issues."""

    # Generate basic prompts (for reference/testing)
    prompts = [template.format(cwes[i], titles[i], descriptions[i]) for i in range(0, len(cwes))]
    print(f"Sample prompt: {prompts[100]}")
    print(f"Total prompts generated: {len(prompts)}")

    # Create simple prompts set with structured output format

    template_1 = """The following is a security issue found in Python: '{}'. {}
    Based on this context and your knowledge about code and its vulnerabilities, generate the following:
    1. Vulnerable Python code of 20-50 lines containing this security issue.
    2. Reasoning on why this code is vulnerable.
    3. Secure/fixed version of this code with minimal change.
    4. Reasoning on why the second version is fixed.
    5. A short instruction summary (one or two
    sentence) that can be used by a language model to generate either of the code.

    There should be no comment in the code. Reasoning should be within one or two sentence. 
    The security issue should be implicit. The issue should not be mentioned in the instruction summary. 
    Give your answer in following format:
    VUL: <vulnerable code>
    REASON: <reasoning for vulnerable code>
    SEC: <secure code>
    REASON: <reasoning for secure code>
    INST: <instruction summary>"""

    # Generate simple prompts using structured template
    prompts_set_simple = [template_1.format(titles[i], descriptions[i]) for i in range(0, len(cwes))]
    print(f"Sample simple prompt: {prompts_set_simple[3]}")

    # Save simple prompts to CSV
    prompts_set_simple_df = pd.DataFrame({"Prompt": prompts_set_simple})
    prompts_set_simple_df.to_csv("#SIMPLE_PROMPTS_FILEPATH")


    # Create complex prompts set with random module selection

    template_2 = """The following is a security issue found in Python: '{}'. {}
    Based on this context and your knowledge about code and its vulnerabilities, generate the following:
    1. Vulnerable {} Python code of 20-50 lines containing this security issue.
    2. Reasoning on why this code is vulnerable.
    3. Secure/fixed version of this code with minimal change.
    4. Reasoning on why the second version is fixed.
    5. A short instruction summary (one or two
    sentence) that can be used by a language model to generate either of the code.

    There should be no comment in the code. Reasoning should be within one or two sentence. 
    The security issue should be implicit. The issue should not be mentioned in the instruction summary. 
    Give your answer in following format:
    VUL: <vulnerable code>
    REASON: <reasoning for vulnerable code>
    SEC: <secure code>
    REASON: <reasoning for secure code>
    INST: <instruction summary>"""

    # Configuration for complex prompt generation
    number_of_examples = 10000  # Total number of complex prompts to generate
    number_of_divisions = 500   # For batch processing (if needed)

    prompts_set_complex = []

    # Create index lists for random sampling
    rules_possible_idx = [i for i in range(0, len(cwes))]
    modules_possible_idx = [i for i in range(0, len(modules))]

    # Generate complex prompts by randomly combining rules with modules
    for i in range(number_of_examples):
        rule_idx = random.choice(rules_possible_idx)
        module_idx = random.choice(modules_possible_idx)
        prompts_set_complex.append(template_2.format(titles[rule_idx], descriptions[rule_idx], modules[module_idx]))

    print(f"Sample complex prompt: {prompts_set_complex[30]}")

    # Save complex prompts to CSV
    prompts_set_complex_df = pd.DataFrame({"Prompt": prompts_set_complex})
    prompts_set_complex_df.to_csv("#COMPLEX_PROMPTS_FILEPATH")
    
    print(f"✓ Generated {len(prompts_set_simple)} simple prompts")
    print(f"✓ Generated {len(prompts_set_complex)} complex prompts")
    print("Prompt generation completed successfully!")
