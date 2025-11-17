"""
Script to preprocess FinQA JSON datasets and save as CSV files.
"""
import json
import pandas as pd
from pathlib import Path
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


def preprocess_finqa(base_path: str = "data") -> None:
    """
    Preprocess FinQA JSON datasets and save as CSV files.
    
    Args:
        base_path: Base path to the data directory (default: "data")
    """
    base_path = Path(base_path)
    finqa_path = base_path / "FinQA"
    clean_finqa_path = base_path / "clean"
    output_path = clean_finqa_path / "FinQA"
    
    if not finqa_path.exists():
        raise FileNotFoundError(f"FinQA directory not found at {finqa_path}")
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    finqa_files = {
        "train": "train.json",
        "dev": "dev.json",
        "test": "test.json"
    }
    
    for key, filename in finqa_files.items():
        file_path = finqa_path / filename
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
                        "question": safe_str(qa_section.get("question")),
                        "answer": safe_str(qa_section.get("exe_ans")) if safe_str(qa_section.get("exe_ans")) != "" else safe_str(qa_section.get("answer")),
                        "program": safe_str(qa_section.get("program")),
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
    # Preprocess FinQA datasets and save as CSV
    preprocess_finqa()
    print("\nPreprocessing complete!")

