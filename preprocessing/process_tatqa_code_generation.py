"""
Script to process TAT-QA data and generate code using OpenAI API.
Supports async processing for faster inference.
"""
import os
import sys
import asyncio
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from openai import AsyncOpenAI

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from .utils import (
        initialize_openai_client,
        process_single_question,
        process_single_question_async,
        get_openai_api_key
    )
except ImportError:
    from utils import (
        initialize_openai_client,
        process_single_question,
        process_single_question_async,
        get_openai_api_key  
    )
from config.generation_config import CodeGenerationConfig
from config.prompt_templates import four_shot_prompt_tatqa_templates, build_finetuning_prompt


def combine_context(table: str, paragraphs: str) -> str:
    """
    Combine table and paragraphs into a single context string for TAT-QA.
    
    Args:
        table: Table content
        paragraphs: Paragraph text content
        
    Returns:
        Combined context string
    """
    parts = []
    if paragraphs and str(paragraphs).strip():
        parts.append(str(paragraphs).strip())
    if table and str(table).strip():
        parts.append(str(table).strip())
    
    return "\n\n".join(parts)


async def process_samples_async(
    df: pd.DataFrame,
    config: CodeGenerationConfig,
    max_concurrent: int = 10
):
    """
    Process samples asynchronously for faster inference.
    
    Args:
        df: DataFrame with samples to process
        config: CodeGenerationConfig instance
        max_concurrent: Maximum number of concurrent API requests
        
    Returns:
        List of result dictionaries
    """
    # Initialize async OpenAI client
    print(f"Initializing async OpenAI client (max_concurrent={max_concurrent})...")
    api_key = get_openai_api_key()
    async_client = AsyncOpenAI(api_key=api_key)
    
    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Prepare tasks
    tasks = []
    for idx, row in df.iterrows():
        context = combine_context(
            row.get('table', ''),
            row.get('paragraphs', '')
        )
        question = str(row.get('question', ''))
        answer = str(row.get('answer', ''))
        derivation = str(row.get('derivation', ''))
        
        # Build fine-tuning prompt before creating task
        
        task = process_single_question_async(
            async_client=async_client,
            context=context,
            question=question,
            answer=answer,
            program=derivation,  # Use derivation as program for TAT-QA
            few_shot_examples=four_shot_prompt_tatqa_templates,
            config=config,
            semaphore=semaphore,
            conversation_history=None  # TAT-QA doesn't have conversation history
        )
        tasks.append(task)
    
    # Process all tasks with progress bar
    print("Processing samples asynchronously...")
    results = []
    
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing samples asynchronously"):
        result = await coro
        
        # Extract prompt from result
        finetuning_prompt = build_finetuning_prompt(result.get('context', ''), result.get('question', ''))
        
        # Only include essential fields in final output
        results.append({
            'prompt': finetuning_prompt,
            'generated_code': result.get('extracted_code', ''),
            'actual_answer': result.get('actual_answer', ''),
            'expected_answer': result.get('expected_answer', '')
        })
    
    # Close async client
    await async_client.close()
    
    return results


def process_tatqa_samples(
    input_csv: str,
    output_csv: str,
    config: CodeGenerationConfig = None,
    max_concurrent: int = 10,
    use_async: bool = True
):
    """
    Process TAT-QA CSV and generate code.
    
    Args:
        input_csv: Path to input CSV file
        output_csv: Path to output CSV file
        config: CodeGenerationConfig instance
        max_concurrent: Maximum number of concurrent API requests (for async mode)
        use_async: Whether to use async processing (faster) or sequential processing
    """
    # Read CSV file
    print(f"\nReading {input_csv}...")
    df = pd.read_csv(input_csv)
    print(f"✓ Loaded {len(df)} samples")
    
    if use_async:
        # Use async processing
        results = asyncio.run(process_samples_async(df, config, max_concurrent))
    else:
        # Use sequential processing (original)
        print(f"Initializing OpenAI client...")
        try:
            client = initialize_openai_client(config=config)
            print("✓ Client initialized successfully")
        except Exception as e:
            print(f"✗ Error initializing client: {e}")
            return
        
        results = []
        for idx, row in tqdm(df.iterrows(), desc="Processing samples", total=len(df)):
            context = combine_context(
                row.get('table', ''),
                row.get('paragraphs', '')
            )
            question = str(row.get('question', ''))
            answer = str(row.get('answer', ''))
            derivation = str(row.get('derivation', ''))
            
            result = process_single_question(
                client=client,
                context=context,
                question=question,
                answer=answer,
                program=derivation,  # Use derivation as program for TAT-QA
                few_shot_examples=four_shot_prompt_tatqa_templates,
                conversation_history=None,  # TAT-QA doesn't have conversation history
                config=config
            )
            
            finetuning_prompt = build_finetuning_prompt(context, question)
            generated_code = result.get('extracted_code', '')
            actual_answer = result.get('actual_answer', '')
            
            result_row = {
                'prompt': finetuning_prompt,
                'generated_code': generated_code,
                'actual_answer': actual_answer,
                'expected_answer': answer
            }
            
            results.append(result_row)
    
    # Create output DataFrame
    output_df = pd.DataFrame(results)
    
    # Create output directory if it doesn't exist
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    output_df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"\n✓ Saved {len(output_df)} results to {output_csv}")
    
    # Print summary
    print("\n" + "="*60)
    print("Summary:")
    print("="*60)
    print(f"Total processed: {len(results)}")
    print(f"Successfully generated code: {sum(1 for r in results if r.get('generated_code', '').strip())}")
    print(f"Missing generated code: {sum(1 for r in results if not r.get('generated_code', '').strip())}")


if __name__ == "__main__":
    # Create configuration
    config = CodeGenerationConfig(
        model_name="gpt-5",  
        max_completion_tokens=5000
    )
    
    # Process samples with async processing (faster)
    # Adjust max_concurrent based on your API rate limits
    # Higher values = faster but may hit rate limits
    process_tatqa_samples(
        input_csv="data/clean/TAT-QA/train.csv",
        output_csv="data/clean_with_code/TAT-QA/tatqa_train_generated.csv",
        config=config,
        max_concurrent=10,  # Adjust based on API rate limits (try 5-20)
        use_async=True  # Set to False for sequential processing
    )

    process_tatqa_samples(
        input_csv="data/clean/TAT-QA/dev.csv",
        output_csv="data/clean_with_code/TAT-QA/tatqa_dev_generated.csv",
        config=config,
        max_concurrent=10,  # Adjust based on API rate limits (try 5-20)
        use_async=True  # Set to False for sequential processing
    )

    process_tatqa_samples(
        input_csv="data/clean/TAT-QA/test.csv",
        output_csv="data/clean_with_code/TAT-QA/tatqa_test_generated.csv",
        config=config,  
        max_concurrent=10,  # Adjust based on API rate limits (try 5-20)
        use_async=True  # Set to False for sequential processing
    )