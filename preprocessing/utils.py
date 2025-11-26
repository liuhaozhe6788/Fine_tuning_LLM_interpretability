import re
import pandas as pd
import os
import json
import sys
import asyncio
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from openai import OpenAI, AsyncOpenAI
from config.generation_config import CodeGenerationConfig, default_config
from config.prompt_templates import build_teacher_prompt, build_teacher_prompt_with_conversation_history


def get_openai_api_key() -> str:
    """
    Get OpenAI API key from environment variable.
    
    Returns:
        API key string
        
    Raises:
        ValueError: If API key is not found
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found in environment variables. "
            "Please set it in your .env file or export it as an environment variable."
        )
    return api_key


def initialize_openai_client(config: Optional[CodeGenerationConfig] = None) -> OpenAI:
    """
    Initialize OpenAI client using configuration.
    
    Args:
        config: CodeGenerationConfig instance. If None, uses default_config.
        
    Returns:
        Configured OpenAI client instance
    """
    if config is None:
        config = default_config
    
    api_key = get_openai_api_key()
    
    client = OpenAI(api_key=api_key)
    
    # Store config in client for later use
    client.config = config
    
    return client


def build_few_shot_teacher_prompt(
    few_shot_examples: list,
    context: str,
    question: str,
    program: str,
    conversation_history: str = None
) -> str:
    """
    Build a prompt with few-shot examples for teacher code generation.
    
    Args:
        dataset: The dataset name
        few_shot_examples: List of few-shot example strings
        context: The document/table context for the current question
        question: The question to answer
        program: The program to use for code generation
        conversation_history: The previous questions in conversation

    Returns:
        Formatted prompt string
    """   
    # Build the current example prompt
    if conversation_history is not None:
        current_example = build_teacher_prompt_with_conversation_history(context, question, program, conversation_history)
    else:
        current_example = build_teacher_prompt(context, question, program)
    if len(few_shot_examples) > 0:
        # Combine few-shot examples
        few_shot_text = "\n\n".join(few_shot_examples)
        full_prompt = few_shot_text + "\n\n" + current_example
        return full_prompt
    else:
        return current_example


def generate_code(
    client: OpenAI,
    context: str,
    question: str,
    program: str,
    few_shot_examples: list,
    conversation_history: None,
    config: Optional[CodeGenerationConfig] = None
) -> str:
    """
    Generate Python code using OpenAI API with configuration.
    
    Args:
        client: Initialized OpenAI client
        context: Document/table context
        question: Question to answer
        program: Program to use for code generation
        few_shot_examples: List of few-shot example strings
        conversation_history: The previous questions in conversation
        config: CodeGenerationConfig instance. If None, uses client.config or default_config.
        
    Returns:
        Generated Python code string
    """
    # Get config from client or use provided/default
    if config is None:
        config = getattr(client, 'config', default_config)
    
    # Build the prompt 
    prompt = build_few_shot_teacher_prompt(few_shot_examples, context, question, program, conversation_history)
    
    try:
        # Get OpenAI API parameters from config
        api_params = config.get_openai_params()
        
        # Generate response using OpenAI API
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            **api_params
        )
        
        # Extract the generated text
        generated_code = response.choices[0].message.content.strip()
        
        return generated_code
    
    except Exception as e:
        raise RuntimeError(f"Error generating code with OpenAI API: {str(e)}")


def extract_code_from_response(
    response_text: str,
    config: Optional[CodeGenerationConfig] = None
) -> str:
    """
    Extract Python code from the model response.
    
    Args:
        response_text: Raw response from the model
        config: CodeGenerationConfig instance. If None, uses default_config.
        
    Returns:
        Extracted Python code
    """
    if config is None:
        config = default_config
    
    # Look for code between markers
    start_marker = config.code_start_marker
    end_marker = config.code_end_marker
    
    if start_marker in response_text:
        start_idx = response_text.find(start_marker) + len(start_marker)
        if end_marker in response_text:
            end_idx = response_text.find(end_marker)
            code = response_text[start_idx:end_idx].strip()
        else:
            # If no end marker, take everything after start marker
            code = response_text[start_idx:].strip()
        return code
    else:
        # If no markers, return the whole response
        return response_text.strip()


def execute_code(code: str) -> Tuple[Any, Optional[str]]:
    """
    Execute generated Python code and return the result.
    
    Args:
        code: Python code string to execute
        
    Returns:
        Tuple of (result, error_message). result is the value of 'ans' variable, 
        or None if execution failed. error_message is None if successful.
    """
    try:
        # Create a safe execution environment
        local_vars = {}
        exec(code, {"__builtins__": __builtins__}, local_vars)
        
        # Get the answer from the 'ans' variable
        if "ans" in local_vars:
            return local_vars["ans"], None
        else:
            return None, "Variable 'ans' not found in generated code"
    
    except Exception as e:
        return None, str(e)


def process_single_question(
    client: OpenAI,
    context: str,
    question: str,
    answer: str,
    program: str,
    few_shot_examples: list,
    conversation_history: str = None,
    config: Optional[CodeGenerationConfig] = None
) -> Dict[str, Any]:
    """
    Process a single question: generate code.
    
    Args:
        client: Initialized OpenAI client
        context: Document/table context
        question: Question to answer
        answer: Expected answer
        program: Program to use for code generation
        few_shot_examples: List of few-shot examples
        conversation_history: The previous questions in conversation
        config: CodeGenerationConfig instance. If None, uses client.config or default_config.
        
    Returns:
        Dictionary with generation results
    """
    # Get config from client or use provided/default
    if config is None:
        config = getattr(client, 'config', default_config)
    
    result = {
        "question": question,
        "expected_answer": answer,
        "generated_code": None,
        "extracted_code": None,
        "actual_answer": None,
        "error": None
    }
    
    try:
        # Generate code
        generated_code = generate_code(
            client=client,
            context=context,
            question=question,
            program=program,
            few_shot_examples=few_shot_examples,
            conversation_history=conversation_history,
            config=config
        )
        
        result["generated_code"] = generated_code
        result["extracted_code"] = extract_code_from_response(generated_code, config)
        
        if result["extracted_code"]:
            actual_answer, error = execute_code(result["extracted_code"])
            result["actual_answer"] = actual_answer
            if error:
                result["error"] = error
        
        return result
    
    except Exception as e:
        result["error"] = str(e)
        return result

async def process_single_question_async(
    async_client: AsyncOpenAI,
    context: str,
    question: str,
    answer: str,
    program: str,
    few_shot_examples: list,
    config: CodeGenerationConfig,
    semaphore: asyncio.Semaphore,
    conversation_history: str = None,
):
    """
    Async version of process_single_question for parallel processing.
    
    Args:
        async_client: Async OpenAI client
        context: Document/table context
        question: Question to answer
        answer: Expected answer
        program: Program to use for code generation
        few_shot_examples: List of few-shot examples
        config: CodeGenerationConfig instance
        semaphore: Semaphore to limit concurrent requests
        conversation_history: The previous questions in conversation

    Returns:
        Dictionary with generation results
    """
    async with semaphore:  # Limit concurrent requests
        try:
            
            # Build the prompt
            prompt = build_few_shot_teacher_prompt(few_shot_examples, context, question, program, conversation_history)
            
            # Get OpenAI API parameters from config
            api_params = config.get_openai_params()
            
            # Generate response using async OpenAI API
            response = await async_client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                **api_params
            )
            result = {
                "context": context,
                "conversation_history": conversation_history,
                "question": question,
                "expected_answer": answer,                
                "generated_code": None,
                "extracted_code": None,
                "actual_answer": None,
                "error": None
            }
            # Extract the generated text
            result["generated_code"] = response.choices[0].message.content.strip()
            result["extracted_code"] = extract_code_from_response(result["generated_code"], config)
            
            if result["extracted_code"]:
                actual_answer, error = execute_code(result["extracted_code"])
                result["actual_answer"] = actual_answer
                if error:
                    result["error"] = error
            
            return result
        
        except Exception as e:
            result = {
                "context": context,
                "conversation_history": conversation_history,
                "question": question,
                "expected_answer": answer,
                "generated_code": None,
                "extracted_code": None,
                "actual_answer": None,
                "error": str(e)
            }
            return result

def normalize_text_section(section) -> str:
    """
    Normalize a text section (list of strings) into a single formatted string.
    
    Args:
        section: List of strings or a single string
        
    Returns:
        Formatted string with each part on a new line
    """
    if isinstance(section, list):
        parts = [str(item).strip() for item in section if item and str(item).strip()]
        if not parts:
            return ""
        normalized_text = "\n".join(parts)
        return normalized_text
    if section is None:
        return ""
    return str(section).strip()


def safe_str(value) -> str:
    """
    Safely convert a value to a string, handling None values.
    
    Args:
        value: Value to convert to string
        
    Returns:
        String representation of the value, or empty string if None
    """
    if value is None:
        return ""
    return str(value).strip()


def compare_answers(actual: str | int | float | bool, expected: str) -> bool:
    """
    Compare actual and expected answers, handling numeric and string comparisons.
    
    Args:
        actual: Actual answer from code execution
        expected: Expected answer from dataset
        
    Returns:
        True if answers match, False otherwise
    """
    if not (isinstance(actual, str) or isinstance(actual, int) or isinstance(actual, float) or isinstance(actual, bool)):
        return False
    if pd.isna(actual) or actual is None:
        return False
    
    if pd.isna(expected) or expected is None:
        return False
    
    # Convert to strings and strip whitespace
    actual_str = str(actual).strip()
    expected_str = str(expected).strip()
    
    # Try numeric comparison first
    try:
        # Try numeric comparison
        expected_num = float(expected_str)
        result_num = float(actual_str)
        is_valid = abs(expected_num - result_num) < 1e-3
        return is_valid
    except (ValueError, TypeError):
        # Fall back to string comparison
        is_valid = actual_str.lower() == expected_str.lower()
        if is_valid:
            return True
        elif isinstance(actual, bool):
            if actual and expected_str == "yes":
                return True
            elif not actual and expected_str == "no":
                return True
            else:
                return False
        else:
            return False

def format_table(table_section) -> str:
    """
    Format a table section (nested list) into a string with rows separated by newlines
    and columns separated by |.
    
    Args:
        table_section: Nested list representing table rows and cells
        
    Returns:
        Formatted table string
    """
    if not table_section:
        return ""

    lines = []
    for row in table_section:
        if isinstance(row, list):
            cells = [safe_str(cell) for cell in row]
        else:
            cells = [safe_str(row)]
        # Join cells with | separator
        line = "|".join(cells)
        lines.append(line)

    # Join rows with newline separator
    formatted_table = "\n".join(lines)
    return formatted_table

def print_dataset_summary(df: pd.DataFrame, split_name: str) -> None:
    """
    Print a summary of the cleaned dataset.
    
    Args:
        df: DataFrame containing the cleaned data
        split_name: Name of the dataset split (train, dev, test)
    """
    print(f"\n  Summary for {split_name} dataset:")
    print(f"    Total examples: {len(df)}")
    print(f"    Columns: {', '.join(df.columns.tolist())}")
    
    # Count non-empty values for each column
    print(f"\n    Non-empty values per column:")
    for col in df.columns:
        non_empty = df[col].astype(str).str.strip().ne('').sum()
        percentage = (non_empty / len(df)) * 100 if len(df) > 0 else 0
        print(f"      {col}: {non_empty}/{len(df)} ({percentage:.1f}%)")
    
    # Calculate average text lengths for text columns
    print(f"\n    Average text lengths:")
    for col in df.columns:
        if col in df.columns:
            lengths = df[col].astype(str).str.len()
            avg_len = lengths.mean()
            median_len = lengths.median()
            max_len = lengths.max()
            print(f"      {col}: avg={avg_len:.1f}, median={median_len:.1f}, max={max_len}")


def main():
    """
    Debug/test function to run process_single_question with sample data.
    """
    from config.prompt_templates import four_shot_prompt_finqa_templates
    
    # Create configuration
    config = CodeGenerationConfig(
        model_name="gpt-5",
        max_completion_tokens=5000
    )
    
    # Initialize the client
    print("="*60)
    print(f"\nInitializing OpenAI client ({config.model_name})...")
    try:
        client = initialize_openai_client(config=config)
        print("✓ Client initialized successfully")
    except Exception as e:
        print(f"✗ Error initializing client: {e}")
        return
    
    # Sample test data
    test_context = """
    17. Income Taxes
    Income before income taxes for the Company's domestic and foreign operations was as follows:
    — | — | Years Ended June 30, | —
    ($ in millions) | 2019 | 2018 | 2017
    Domestic | $204.2 | $140.3 | $56.0
    Foreign | 11.8 | 19.9 | 14.2
    Income before income taxes | $216.0 | $160.2 | $70.2
    """
    
    test_question = "What was the change in Foreign in 2019 from 2018?"
    test_answer = "-8.1"
    test_program = "subtract(11.8, 19.9)"
    
    print(f"\nTest Question: {test_question}")
    print(f"Expected Answer: {test_answer}")
    print(f"Program: {test_program}")
    print(f"\nProcessing...")
    
    # Process the question
    result = process_single_question(
        client=client,
        context=test_context,
        question=test_question,
        answer=test_answer,
        program=test_program,
        few_shot_examples=four_shot_prompt_finqa_templates,
        config=config
    )
    
    # Print results
    print("\n" + "="*60)
    print("Results:")
    print("="*60)
    print(f"\nQuestion: {result['question']}")
    print(f"Expected Answer: {result['expected_answer']}")
    
    if result.get('generated_code'):
        print(f"\nGenerated Code (full response):")
        print("-" * 60)
        print(result['generated_code'][:500] + "..." if len(result['generated_code']) > 500 else result['generated_code'])
        print("-" * 60)
    
    if result.get('extracted_code'):
        print(f"\nExtracted Code:")
        print("-" * 60)
        print(result['extracted_code'])
        print("-" * 60)
    
    if result.get('actual_answer') is not None:
        print(f"\nActual Answer (execution result): {result['actual_answer']}")
    
    if result.get('error'):
        print(f"\nError: {result['error']}")


if __name__ == "__main__":
    main()

