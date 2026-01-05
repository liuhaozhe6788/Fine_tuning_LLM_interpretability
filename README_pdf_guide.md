# Using PDFs for Context with Questions

---

## Prerequisites

```bash
# Install PDF extraction library
pip install pymupdf --break-system-packages

# Uninstall bitsandbytes if you had issues
pip uninstall bitsandbytes -y
```
---

### Step 1: Extract text from your PDFs

```bash
# Single PDF
python extract_pdf_text.py --pdf apple_10k_2025.pdf --output apple_10k_2025.txt --max_pages 50
```

### Step 2: Create your queries JSON file

**Edit the template script**

```bash
# Edit create_queries_json.py (with questions and pdf to point to) then run:
python create_queries_json.py --output queries.json --mode extracted_text
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
```

---