"""
Script to preprocess TAT-QA JSON datasets and save as CSV files.
"""
import json
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
try:
    from .utils import (
        safe_str,
        format_table,
        normalize_text_section,
        print_dataset_summary
    )
except ImportError:
    from utils import (
        safe_str,
        format_table,
        normalize_text_section,
        print_dataset_summary
    )


def extract_table(entry) -> str:
    """
    Extract and format table using utility function.
    
    Args:
        entry: Dictionary containing the data entry
        
    Returns:
        Formatted table string (pipe-separated), or empty string if not available
    """
    table_data = entry.get("table", {})
    if isinstance(table_data, dict):
        table = table_data.get("table", [])
        if isinstance(table, list):
            return format_table(table)
    return ""


def extract_paragraphs(entry) -> str:
    """
    Extract and format paragraphs using utility function.
    
    Args:
        entry: Dictionary containing the data entry
        
    Returns:
        Formatted paragraph text string (newline-separated), or empty string if not available
    """
    paragraphs = entry.get("paragraphs", [])
    if not isinstance(paragraphs, list):
        return ""
    
    # Extract text from each paragraph
    text_list = []
    for para in paragraphs:
        if isinstance(para, dict):
            text = para.get("text", "")
            if text:
                text_list.append(safe_str(text))
    
    # Use normalize_text_section to format as newline-separated string
    return normalize_text_section(text_list)


def extract_qa_derivations(entry) -> list:
    """
    Extract multiple question/answer/derivation as a list of tuples.
    
    Args:
        entry: Dictionary containing the data entry
        
    Returns:
        List of tuples: [(question, answer, derivation), ...]
    """
    questions = entry.get("questions", [])
    if not isinstance(questions, list):
        return []
    
    qa_list = []
    for q in questions:
        if isinstance(q, dict):
            question = safe_str(q.get("question", ""))
            answer = q.get("answer", "")
            derivation = safe_str(q.get("derivation", ""))
            answer_type = safe_str(q.get("answer_type", ""))
            
            if answer_type == "arithmetic":
                # Convert answer to string representation
                if isinstance(answer, list):
                    # Join list items with semicolon or keep as list representation
                    answer_str = "\n".join([safe_str(item) for item in answer])
                else:
                    answer_str = safe_str(answer)
                
                if question:  # Only include if question is not empty
                    qa_list.append((question, answer_str, derivation))
    
    return qa_list


def process_tatqa_file(file_path: Path) -> pd.DataFrame:
    """
    Process a TAT-QA JSON file and return a cleaned DataFrame.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        DataFrame with processed data
    """
    # Load JSON file
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    filtered_data = []
    for entry in raw_data:
        # Extract and format table using utility function
        table = extract_table(entry)
        
        # Extract and format paragraphs using utility function
        paragraphs = extract_paragraphs(entry)
        
        # Extract question/answer/derivation as list of tuples
        qa_derivations = extract_qa_derivations(entry)
        
        # Create one entry per question
        for question, answer, derivation in qa_derivations:
            filtered_entry = {
                "table": table,  # Formatted table string
                "paragraphs": paragraphs,  # Formatted paragraph text string
                "question": question,
                "answer": answer,
                "derivation": derivation,
            }
            filtered_data.append(filtered_entry)
    
    # Convert to DataFrame
    df = pd.DataFrame(filtered_data)
    
    # Filter out rows with empty questions or answers
    initial_count = len(df)
    df = df[
        df['question'].astype(str).str.strip().ne('') & 
        df['answer'].astype(str).str.strip().ne('')
    ].reset_index(drop=True)
    removed_count = initial_count - len(df)
    
    if removed_count > 0:
        print(f"  Removed {removed_count} rows with empty questions or answers (kept {len(df)} rows)")
    
    return df


def preprocess_tatqa(base_path: str = "data") -> None:
    """
    Preprocess TAT-QA JSON datasets and save as CSV files.
    
    Args:
        base_path: Base path to the data directory (default: "data")
    """
    base_path = Path(base_path)
    tatqa_path = base_path / "TAT-QA"
    clean_tatqa_path = base_path / "clean"
    output_path = clean_tatqa_path / "TAT-QA"
    
    if not tatqa_path.exists():
        raise FileNotFoundError(f"TAT-QA directory not found at {tatqa_path}")
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Define files to process: (filename, output_name, should_split)
    files_to_process = [
        ("tatqa_dataset_train.json", "train", True),
        ("tatqa_dataset_dev.json", "test", False),
    ]
    
    for filename, output_name, should_split in files_to_process:
        file_path = tatqa_path / filename
        if file_path.exists():
            print(f"Processing {file_path}...")
            try:
                # Process the file
                df = process_tatqa_file(file_path)
                
                if should_split:
                    # Split into train and dev sets with 9:1 ratio
                    print(f"\n  Splitting data into train (90%) and dev (10%) sets...")
                    df_train, df_dev = train_test_split(
                        df, 
                        test_size=550, 
                        random_state=42, 
                        shuffle=True
                    )
                    df_train = df_train.reset_index(drop=True)
                    df_dev = df_dev.reset_index(drop=True)
                    
                    print(f"  Train set: {len(df_train)} examples")
                    print(f"  Dev set: {len(df_dev)} examples")
                    
                    # Print summary for train set
                    print_dataset_summary(df_train, "train")
                    
                    # Print summary for dev set
                    print_dataset_summary(df_dev, "dev")
                    
                    # Save train set
                    train_output_file = output_path / "train.csv"
                    df_train.to_csv(train_output_file, index=False, encoding='utf-8')
                    print(f"  Saved {len(df_train)} examples to {train_output_file}")
                    
                    # Save dev set
                    dev_output_file = output_path / "dev.csv"
                    df_dev.to_csv(dev_output_file, index=False, encoding='utf-8')
                    print(f"  Saved {len(df_dev)} examples to {dev_output_file}")
                else:
                    # Print summary before saving
                    print_dataset_summary(df, output_name)
                    
                    # Save as CSV
                    output_file = output_path / f"{output_name}.csv"
                    df.to_csv(output_file, index=False, encoding='utf-8')
                    print(f"  Saved {len(df)} examples to {output_file}")
                    
            except Exception as e:
                print(f"  Error processing {file_path}: {e}")
        else:
            print(f"  Warning: {file_path} not found")


if __name__ == "__main__":
    # Preprocess TAT-QA datasets and save as CSV
    preprocess_tatqa()
    print("\nPreprocessing complete!")
