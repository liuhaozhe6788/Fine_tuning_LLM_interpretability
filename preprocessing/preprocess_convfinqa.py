"""
Script to preprocess ConvFinQA JSON datasets and save as CSV files.
"""
import json
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
try:
    from .utils import (
        normalize_text_section,
        safe_str,
        format_table,
        print_dataset_summary
    )
except ImportError:
    from utils import (
        normalize_text_section,
        safe_str,
        format_table,
        print_dataset_summary
    )


def extract_last_question(entry) -> str:
    """
    Extract the last question from dialogue_break.
    
    Args:
        entry: Dictionary containing the data entry
        
    Returns:
        The last question from dialogue_break, or empty string if not available
    """
    annotation = entry.get("annotation", {}) or {}
    dialogue_break = annotation.get("dialogue_break", [])
    
    if not dialogue_break or not isinstance(dialogue_break, list) or len(dialogue_break) == 0:
        return ""
    
    # Get the last question
    last_question = dialogue_break[-1]
    return safe_str(last_question)


def extract_last_answer(entry) -> str:
    """
    Extract the last answer from exe_ans_list.
    
    Args:
        entry: Dictionary containing the data entry
        
    Returns:
        The last answer from exe_ans_list as a string, or empty string if not available
    """
    annotation = entry.get("annotation", {}) or {}
    exe_ans_list = annotation.get("exe_ans_list", [])
    
    if not exe_ans_list or not isinstance(exe_ans_list, list) or len(exe_ans_list) == 0:
        return ""
    
    # Get the last answer
    last_answer = exe_ans_list[-1]
    return safe_str(last_answer)


def extract_last_program(entry) -> str:
    """
    Extract the last program from turn_program.
    
    Args:
        entry: Dictionary containing the data entry
        
    Returns:
        The last program from turn_program as a string, or empty string if not available
    """
    annotation = entry.get("annotation", {}) or {}
    turn_program = annotation.get("turn_program", [])
    
    if not turn_program or not isinstance(turn_program, list) or len(turn_program) == 0:
        return ""
    
    # Get the last program
    last_program = turn_program[-1]
    return safe_str(last_program)


def extract_conversation_history(entry) -> str:
    """
    Extract the previous turns of questions from dialogue_break.
    
    Args:
        entry: Dictionary containing the data entry
        
    Returns:
        Formatted string with the previous questions, or empty string if not available
    """
    annotation = entry.get("annotation", {}) or {}
    dialogue_break = annotation.get("dialogue_break", [])
    
    if not dialogue_break or not isinstance(dialogue_break, list):
        return ""
    
    # Get all questions except the last one (current question)
    conversation_history = dialogue_break[:-1] if len(dialogue_break) > 1 else []

    if not conversation_history:
        return ""
    
    # Format as a conversation history string
    # Each question on a new line with a prefix
    formatted_history = "\n".join(conversation_history)
    return formatted_history


def preprocess_convfinqa(base_path: str = "data") -> None:
    """
    Preprocess ConvFinQA JSON datasets and save as CSV files.
    
    Args:
        base_path: Base path to the data directory (default: "data")
    """
    base_path = Path(base_path)
    convfinqa_path = base_path / "ConvFinQA"
    clean_convfinqa_path = base_path / "clean"
    output_path = clean_convfinqa_path / "ConvFinQA"
    
    if not convfinqa_path.exists():
        raise FileNotFoundError(f"ConvFinQA directory not found at {convfinqa_path}")
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    convfinqa_files = {
        "train": "train.json",
        "test": "dev.json"
    }
    
    for key, filename in convfinqa_files.items():
        file_path = convfinqa_path / filename
        if file_path.exists():
            print(f"Processing {file_path}...")
            try:
                # Load JSON file
                with open(file_path, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)

                filtered_data = []
                for entry in raw_data:
                    qa_section = entry.get("qa", {}) or {}
                    filtered_entry = {
                        "pre_text": normalize_text_section(entry.get("pre_text", [])),
                        "post_text": normalize_text_section(entry.get("post_text", [])),
                        "table": format_table(entry.get("table", [])),
                        "question": extract_last_question(entry),
                        "answer": extract_last_answer(entry),
                        "program": extract_last_program(entry),
                        "conversation_history": extract_conversation_history(entry),
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
                
                # Split into train and dev sets with 9:1 ratio
                if key == "train":
                    print(f"\n  Splitting data into train (90%) and dev (10%) sets...")
                    df_train, df_dev = train_test_split(
                        df, 
                        test_size=300, 
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
                
                elif key == "test":
                    # Print summary before saving
                    print_dataset_summary(df, key)
                    
                    # Save as CSV
                    output_file = output_path / f"{key}.csv"
                    df.to_csv(output_file, index=False, encoding='utf-8')
                    print(f"  Saved {len(df)} examples to {output_file}")
            except Exception as e:
                print(f"  Error processing {file_path}: {e}")
        else:
            print(f"  Warning: {file_path} not found")


if __name__ == "__main__":
    # Preprocess ConvFinQA datasets and save as CSV
    preprocess_convfinqa()
    print("\nPreprocessing complete!")

