# Local Document AI — Jupyter Notebook Solution

Classify PDFs, extract structured data, and search by meaning.
**No paid APIs. Runs 100% locally.**

---

## What's included

```
document_ai.ipynb   ← Main notebook (open this)
output.json         ← Generated results
sample_docs/        ← 12 sample PDF documents
README.md           ← This file
```

---

## How to run

### 1. Install Python libraries
```bash
pip install pdfminer.six scikit-learn sentence-transformers faiss-cpu jupyter
```

### 2. Open the notebook
```bash
jupyter notebook document_ai.ipynb
```

### 3. Run all cells top to bottom
`Cell → Run All`

That's it. `output.json` will be created automatically.

---

## What the notebook does

| Step | Description |
|------|-------------|
| **Step 0** | Install all libraries |
| **Step 1** | Read every PDF from `sample_docs/` folder |
| **Step 2** | Classify each document (Invoice / Resume / Utility Bill / Other) |
| **Step 3** | Extract structured fields (invoice number, name, kWh, etc.) |
| **Step 4** | Save results to `output.json` |
| **Step 5** | Search documents by meaning (e.g. "payments due in January") |
| **Bonus**  | Better search using SentenceTransformers + FAISS (optional) |

---

## Libraries used

| Library | Why |
|---------|-----|
| `pdfminer.six` | Read text from PDF files |
| `scikit-learn` | TF-IDF vectors for classifying and searching |
| `sentence-transformers` | Better semantic search (optional, needs ~80 MB download once) |
| `faiss-cpu` | Fast vector search index |
| `re` / `json` | Built-in Python — regex extraction and JSON output |

---
