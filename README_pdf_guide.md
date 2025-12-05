# Step-by-Step Guide: Using PDFs for KL Divergence Questions

This guide shows you how to use PDF documents (like Apple 10-K filings) as context for your KL divergence questions.

---

## Prerequisites

```bash
# Install PDF extraction library
pip install pymupdf --break-system-packages

# Uninstall bitsandbytes if you had issues
pip uninstall bitsandbytes -y
```

---

## Method 1: Extract PDF → Text File → Use in Questions (RECOMMENDED)

This method is more memory-efficient and gives you more control.

### Step 1: Extract text from your PDFs

```bash
# Single PDF
python extract_pdf_text.py --pdf apple_10k_2025.pdf --output apple_10k_2025.txt

# With page limit (for large PDFs)
python extract_pdf_text.py --pdf apple_10k_2025.pdf --output apple_10k_2025.txt --max_pages 50

# Multiple PDFs in a directory
python extract_pdf_text.py --pdf_dir ./pdfs/ --output_dir ./extracted_texts/

# Save as JSON (includes page-by-page breakdown)
python extract_pdf_text.py --pdf apple_10k_2025.pdf --output apple_10k_2025.json --format json
```

### Step 2: Create your queries JSON file

**Option A: Edit the template script**

```bash
# Edit create_queries_json.py to add your questions, then run:
python create_queries_json.py --output queries.json --mode extracted_text
```

**Option B: Create manually**

Create a file called `queries.json`:

```json
{
  "queries": [
    {
      "question": "What are Apple's main risk factors related to supply chain?",
      "documents": [
        {"type": "txt", "path": "apple_10k_2025.txt"}
      ]
    },
    {
      "question": "How did tariffs affect Apple's gross margins in 2025?",
      "documents": [
        {"type": "txt", "path": "apple_10k_2025.txt"}
      ]
    },
    {
      "question": "Compare Apple's financial performance between 2019 and 2025.",
      "documents": [
        {"type": "txt", "path": "apple_10k_2019.txt"},
        {"type": "txt", "path": "apple_10k_2025.txt"}
      ]
    }
  ]
}
```

### Step 3: Run KL divergence computation

```bash
python compute_kl_divergence.py \
    --queries_json queries.json \
    --policy_device cuda:0 \
    --ref_device cuda:1 \
    --batch_size 1 \
    --max_new_tokens 256 \
    --max_context_chars 10000 \
    --output_dir results
```

### Step 4: View results

```bash
# View summary
cat results/kl_divergence_*/summary.json

# View full report
cat results/kl_divergence_*/report.txt

# Or load in Python
python -c "
import json
import glob
latest = sorted(glob.glob('results/kl_divergence_*'))[-1]
with open(f'{latest}/summary.json') as f:
    print(json.dumps(json.load(f), indent=2))
"
```

---

## Method 2: Use PDF Directly (Simpler but uses more memory)

### Step 1: Create queries.json with PDF paths

```json
{
  "queries": [
    {
      "question": "What are Apple's main risk factors?",
      "documents": [
        {"type": "pdf", "path": "apple_10k_2025.pdf", "max_pages": 30}
      ]
    }
  ]
}
```

### Step 2: Run computation

```bash
python compute_kl_divergence.py \
    --queries_json queries.json \
    --policy_device cuda:0 \
    --ref_device cuda:1 \
    --batch_size 1 \
    --max_context_chars 8000
```

---

## Document Types Reference

| Type | JSON Format | Use Case |
|------|------------|----------|
| Extracted text file | `{"type": "txt", "path": "doc.txt"}` | Best for large PDFs, pre-processed |
| PDF directly | `{"type": "pdf", "path": "doc.pdf", "max_pages": 30}` | Quick testing, small PDFs |
| Inline text | `{"type": "text", "text": "Your text here..."}` | Short snippets |

---

## Tips

1. **Memory Management**: Use `--max_context_chars 8000` to limit context size (Mistral-7B has ~8K token limit)

2. **Batch Size**: Use `--batch_size 1` for long contexts to avoid OOM

3. **Page Limits**: For large PDFs, use `--max_pages 30` or extract only relevant sections

4. **Pre-extract Text**: Method 1 (extract first) is better because:
   - You can review/edit the extracted text
   - Faster subsequent runs
   - More control over what's included

5. **File Paths**: Paths in queries.json are relative to the JSON file's location

---

## Complete Example Workflow

```bash
# 1. Setup
pip install pymupdf --break-system-packages
pip uninstall bitsandbytes -y

# 2. Extract PDF text
python extract_pdf_text.py --pdf apple_10k_2025.pdf --output apple_10k_2025.txt --max_pages 50

# 3. Create queries (edit create_queries_json.py first with your questions)
python create_queries_json.py --output my_queries.json --mode extracted_text

# 4. Run KL divergence
python compute_kl_divergence.py \
    --queries_json my_queries.json \
    --policy_device cuda:0 \
    --ref_device cuda:1 \
    --batch_size 1 \
    --max_new_tokens 256 \
    --output_dir results

# 5. Check results
cat results/kl_divergence_*/report.txt
```

---

## Troubleshooting

**"No PDF library found"**
```bash
pip install pymupdf --break-system-packages
```

**"CUDA out of memory"**
- Reduce `--max_context_chars` (try 6000)
- Reduce `--max_pages` when extracting
- Use `--batch_size 1`

**"bitsandbytes error"**
```bash
pip uninstall bitsandbytes -y
```

**Path not found errors**
- Use relative paths from the queries.json file location
- Or use absolute paths: `/workspace/Fine_tuning_LLM_interpretability/apple_10k.txt`