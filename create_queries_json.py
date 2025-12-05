"""
Create Queries JSON for KL Divergence Computation
=================================================
This script helps you create a queries.json file that can be used with 
compute_kl_divergence.py

Usage:
    python create_queries_json.py --output queries.json
"""

import json
import os


def create_queries_json(output_path: str, questions: list, document_paths: list = None):
    """
    Create a queries JSON file.
    
    Args:
        output_path: Where to save the JSON file
        questions: List of question dictionaries
        document_paths: Optional list of document paths to attach to all questions
    """
    queries = []
    
    for q in questions:
        query = {
            "question": q["question"],
        }
        
        if "expected_answer" in q:
            query["expected_answer"] = q["expected_answer"]
        
        # Add documents if specified
        if "documents" in q:
            query["documents"] = q["documents"]
        elif document_paths:
            query["documents"] = []
            for doc_path in document_paths:
                if doc_path.endswith('.pdf'):
                    query["documents"].append({"type": "pdf", "path": doc_path})
                elif doc_path.endswith('.txt'):
                    query["documents"].append({"type": "txt", "path": doc_path})
        
        queries.append(query)
    
    output_data = {"queries": queries}
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Created queries file: {output_path}")
    print(f"  - {len(queries)} questions")
    return output_path


# ============== YOUR QUESTIONS GO HERE ==============

# Option 1: Questions WITHOUT document context (model uses its own knowledge)
QUESTIONS_NO_CONTEXT = [
    {
        "question": "What is machine learning?",
    },
    {
        "question": "Explain neural networks.",
    },
]

# Option 2: Questions WITH document context (from extracted PDF text files)
QUESTIONS_WITH_EXTRACTED_TEXT = [
    {
        "question": "What is the condensed, overarching explanation of how Apple's global supply-chain dependencies, exposure to geopolitical and macroeconomic pressures, and vulnerability to cybersecurity and data-security threats collectively form interconnected risks that can materially affect its operations, financial condition, and reputation?",
        "expected_answer": "Apple faces an interlinked network of risks...",
        "documents": [
            {"type": "txt", "path": "apple_10k_2025.txt"}  # Use extracted text file
        ]
    },
    {
        "question": "How did the combination of Apple's 2025 product-release cycle, regional shifts in segment performance, and evolving macroeconomic and tariff conditions collectively shape the company's overall changes in net sales, gross margins, and operating expenses during the fiscal year?",
        "documents": [
            {"type": "txt", "path": "apple_10k_2025.txt"}
        ]
    },
    {
        "question": "How do Apple's 2025 financial disclosures characterize the combined impact of new U.S.-announced tariffs, global macroeconomic pressures, and supply-chain dependencies on its gross margins and overall operational risks?",
        "documents": [
            {"type": "txt", "path": "apple_10k_2025.txt"}
        ]
    },
    {
        "question": "How would you summarize the company's recent financial performance, operating trends, and major risk exposures based on its latest annual disclosures?",
        "documents": [
            {"type": "txt", "path": "apple_10k_2025.txt"}
        ]
    },
    {
        "question": "Based on Apple's 2025 Form 10-K, how did newly introduced U.S. tariffs beginning in Q2 2025 affect Apple's product gross margins, and what related supply-chain and international trade risks does the company identify that could further impact margins in the future?",
        "documents": [
            {"type": "txt", "path": "apple_10k_2025.txt"}
        ]
    },
    {
        "question": "How did the new U.S. tariffs introduced in the second quarter of 2025 affect Apple's 2025 product gross margins, and what risks related to these tariffs does Apple identify that could further impact its future financial performance?",
        "documents": [
            {"type": "txt", "path": "apple_10k_2025.txt"}
        ]
    },
]

# Option 3: Questions comparing TWO documents (2019 vs 2025)
QUESTIONS_MULTI_DOC = [
    {
        "question": "Compare Apple's net income between 2019 and 2025. What was the growth rate?",
        "documents": [
            {"type": "txt", "path": "apple_10k_2019.txt"},
            {"type": "txt", "path": "apple_10k_2025.txt"}
        ]
    },
    {
        "question": "How did Apple's revenue growth strategy change between 2019 and 2025?",
        "documents": [
            {"type": "txt", "path": "apple_10k_2019.txt"},
            {"type": "txt", "path": "apple_10k_2025.txt"}
        ]
    },
]

# Option 4: Questions using PDF directly (requires more memory)
QUESTIONS_WITH_PDF = [
    {
        "question": "What are Apple's main risk factors?",
        "documents": [
            {"type": "pdf", "path": "apple_10k_2025.pdf", "max_pages": 30}
        ]
    },
]


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="queries.json")
    parser.add_argument("--mode", type=str, default="extracted_text",
                        choices=["no_context", "extracted_text", "multi_doc", "pdf"],
                        help="Which question set to use")
    args = parser.parse_args()
    
    if args.mode == "no_context":
        questions = QUESTIONS_NO_CONTEXT
    elif args.mode == "extracted_text":
        questions = QUESTIONS_WITH_EXTRACTED_TEXT
    elif args.mode == "multi_doc":
        questions = QUESTIONS_MULTI_DOC
    elif args.mode == "pdf":
        questions = QUESTIONS_WITH_PDF
    
    create_queries_json(args.output, questions)
    
    print("\nNext step: Run KL divergence computation with:")
    print(f"  python compute_kl_divergence.py --queries_json {args.output}")