"""
PDF Text Extraction Script
==========================
Extract text from PDF files and save to text files or JSON.

Usage:
    python extract_pdf_text.py --pdf path/to/document.pdf --output extracted_text.txt
    python extract_pdf_text.py --pdf path/to/document.pdf --output extracted_text.json --format json
    python extract_pdf_text.py --pdf_dir path/to/pdfs/ --output_dir extracted_texts/
"""

import os
import json
import argparse
from typing import Optional, List, Dict


def extract_text_from_pdf(pdf_path: str, max_pages: Optional[int] = None) -> Dict:
    """
    Extract text from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        max_pages: Maximum number of pages to extract (None for all)
    
    Returns:
        Dictionary with metadata and extracted text
    """
    try:
        import fitz  # PyMuPDF
        
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        pages_to_read = min(total_pages, max_pages) if max_pages else total_pages
        
        pages_text = []
        full_text_parts = []
        
        for page_num in range(pages_to_read):
            page = doc[page_num]
            text = page.get_text()
            pages_text.append({
                "page_number": page_num + 1,
                "text": text,
                "char_count": len(text)
            })
            full_text_parts.append(text)
        
        doc.close()
        
        full_text = "\n\n".join(full_text_parts)
        
        return {
            "source_file": os.path.basename(pdf_path),
            "source_path": os.path.abspath(pdf_path),
            "total_pages": total_pages,
            "pages_extracted": pages_to_read,
            "total_characters": len(full_text),
            "full_text": full_text,
            "pages": pages_text
        }
        
    except ImportError:
        raise ImportError(
            "PyMuPDF not found. Install with:\n"
            "  pip install pymupdf --break-system-packages"
        )


def save_as_text(data: Dict, output_path: str):
    """Save extracted text as a plain text file."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# Source: {data['source_file']}\n")
        f.write(f"# Pages: {data['pages_extracted']} of {data['total_pages']}\n")
        f.write(f"# Characters: {data['total_characters']}\n")
        f.write("=" * 80 + "\n\n")
        f.write(data['full_text'])
    print(f"Saved text to: {output_path}")


def save_as_json(data: Dict, output_path: str):
    """Save extracted text as a JSON file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved JSON to: {output_path}")


def process_single_pdf(pdf_path: str, output_path: str, format: str = "txt", max_pages: Optional[int] = None):
    """Process a single PDF file."""
    print(f"Extracting text from: {pdf_path}")
    
    data = extract_text_from_pdf(pdf_path, max_pages)
    
    print(f"  - Total pages: {data['total_pages']}")
    print(f"  - Pages extracted: {data['pages_extracted']}")
    print(f"  - Characters extracted: {data['total_characters']}")
    
    if format == "json":
        save_as_json(data, output_path)
    else:
        save_as_text(data, output_path)
    
    return data


def process_pdf_directory(pdf_dir: str, output_dir: str, format: str = "txt", max_pages: Optional[int] = None):
    """Process all PDFs in a directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
    print(f"Found {len(pdf_files)} PDF files in {pdf_dir}")
    
    results = []
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        
        # Create output filename
        base_name = os.path.splitext(pdf_file)[0]
        output_ext = ".json" if format == "json" else ".txt"
        output_path = os.path.join(output_dir, base_name + output_ext)
        
        try:
            data = process_single_pdf(pdf_path, output_path, format, max_pages)
            results.append({
                "file": pdf_file,
                "status": "success",
                "pages": data['pages_extracted'],
                "characters": data['total_characters']
            })
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "file": pdf_file,
                "status": "error",
                "error": str(e)
            })
    
    # Save summary
    summary_path = os.path.join(output_dir, "_extraction_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract text from PDF files")
    
    # Single file mode
    parser.add_argument("--pdf", type=str, help="Path to a single PDF file")
    parser.add_argument("--output", type=str, help="Output file path")
    
    # Directory mode
    parser.add_argument("--pdf_dir", type=str, help="Directory containing PDF files")
    parser.add_argument("--output_dir", type=str, help="Output directory for extracted texts")
    
    # Options
    parser.add_argument("--format", type=str, default="txt", choices=["txt", "json"],
                        help="Output format (txt or json)")
    parser.add_argument("--max_pages", type=int, default=None,
                        help="Maximum pages to extract (default: all)")
    
    args = parser.parse_args()
    
    if args.pdf:
        # Single file mode
        if not args.output:
            base_name = os.path.splitext(args.pdf)[0]
            args.output = base_name + (".json" if args.format == "json" else ".txt")
        
        process_single_pdf(args.pdf, args.output, args.format, args.max_pages)
        
    elif args.pdf_dir:
        # Directory mode
        if not args.output_dir:
            args.output_dir = args.pdf_dir + "_extracted"
        
        process_pdf_directory(args.pdf_dir, args.output_dir, args.format, args.max_pages)
        
    else:
        print("Please specify either --pdf for a single file or --pdf_dir for a directory")
        print("\nExamples:")
        print("  python extract_pdf_text.py --pdf apple_10k.pdf --output apple_10k.txt")
        print("  python extract_pdf_text.py --pdf apple_10k.pdf --output apple_10k.json --format json")
        print("  python extract_pdf_text.py --pdf apple_10k.pdf --max_pages 20")
        print("  python extract_pdf_text.py --pdf_dir ./pdfs/ --output_dir ./extracted/")