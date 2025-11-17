"""
Script to filter out instances where actual_answer differs from expected_answer.
Removes mismatched instances and shows statistics.
"""
import pandas as pd
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def compare_answers(actual: str, expected: str) -> bool:
    """
    Compare actual and expected answers, handling numeric and string comparisons.
    
    Args:
        actual: Actual answer from code execution
        expected: Expected answer from dataset
        
    Returns:
        True if answers match, False otherwise
    """
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
        else:
            if actual_str == "True" and expected_str == "yes":
                return True
            elif actual_str == "False" and expected_str == "no":
                return True
            else:
                return False


def filter_valid_answers(
    input_csv: str,
    output_csv: str = None,
    save_filtered: bool = True
):
    """
    Filter out instances where actual_answer differs from expected_answer.
    
    Args:
        input_csv: Path to input CSV file
        output_csv: Path to output CSV file (if None, overwrites input)
        save_filtered: Whether to save the filtered data
    """
    # Read CSV file
    print(f"Reading {input_csv}...")
    df = pd.read_csv(input_csv)
    print(f"✓ Loaded {len(df)} instances")
    
    # Count initial statistics
    total_instances = len(df)
    
    # Count instances with missing actual_answer
    missing_actual_answer = df[df['actual_answer'].isna()]
    missing_actual = df['actual_answer'].isna().sum()
    missing_expected = df['expected_answer'].isna().sum()
    
    # Compare answers
    print("\nComparing actual_answer with expected_answer...")
    df['answers_match'] = df.apply(
        lambda row: compare_answers(row['actual_answer'], row['expected_answer']),
        axis=1
    )
    
    # Count matches and mismatches
    matches = df['answers_match'].sum()
    mismatches = (~df['answers_match']).sum()
    
    # Filter to keep only matching instances
    df_filtered = df[df['answers_match']].copy()
    
    # Remove the temporary column
    df_filtered = df_filtered.drop(columns=['answers_match'])
    
    # Calculate statistics
    removed_count = total_instances - len(df_filtered)
    removal_proportion = (removed_count / total_instances * 100) if total_instances > 0 else 0
    kept_proportion = (len(df_filtered) / total_instances * 100) if total_instances > 0 else 0
    
    # Print statistics
    print("\n" + "="*60)
    print("Filtering Statistics:")
    print("="*60)
    print(f"Total instances: {total_instances}")
    print(f"  - Matching answers: {matches} ({matches/total_instances*100:.2f}%)")
    print(f"  - Mismatched answers: {mismatches} ({mismatches/total_instances*100:.2f}%)")
    print(f"  - Missing actual_answer: {missing_actual} ({missing_actual/total_instances*100:.2f}%)")
    print(f"  - Missing expected_answer: {missing_expected} ({missing_expected/total_instances*100:.2f}%)")
    print("\n" + "-"*60)
    print(f"Instances kept: {len(df_filtered)} ({kept_proportion:.2f}%)")
    print(f"Instances removed: {removed_count} ({removal_proportion:.2f}%)")
    print("="*60)
    
    # Show some examples of mismatches (if any)
    if mismatches > 0:
        print("\nSample mismatches (first 5):")
        print("-"*60)
        mismatched_df = df[~df['answers_match']].head(5)
        for idx, row in mismatched_df.iterrows():
            print(f"  Expected: {row['expected_answer']} | Actual: {row['actual_answer']}")
    
    # Save filtered data
    if save_filtered:
        if output_csv is None:
            output_csv = input_csv
        
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df_filtered.to_csv(output_csv, index=False, encoding='utf-8')
        print(f"\n✓ Saved {len(df_filtered)} filtered instances to {output_csv}")
    else:
        print("\n(Filtered data not saved - set save_filtered=True to save)")
    
    return df_filtered


if __name__ == "__main__":
    # Default paths
    input_csv = "data/clean_with_code/FinQA/finqa_train_generated.csv"
    output_csv = "data/clean_with_code/FinQA/finqa_train_generated_filtered.csv"  
    
    filter_valid_answers(
        input_csv=input_csv,
        output_csv=output_csv,
        save_filtered=True
    )

    input_csv = "data/clean_with_code/FinQA/finqa_dev_generated.csv"
    output_csv = "data/clean_with_code/FinQA/finqa_dev_generated_filtered.csv"  
    
    filter_valid_answers(
        input_csv=input_csv,
        output_csv=output_csv,
        save_filtered=True
    )

    input_csv = "data/clean_with_code/FinQA/finqa_test_generated.csv"
    output_csv = "data/clean_with_code/FinQA/finqa_test_generated_filtered.csv"  

    filter_valid_answers(
        input_csv=input_csv,
        output_csv=output_csv,
        save_filtered=True
    )

