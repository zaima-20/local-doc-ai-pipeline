#!/usr/bin/env python
# coding: utf-8

# In[2]:


import subprocess
subprocess.run(['pip', 'install', 'pdfminer.six', 'scikit-learn', 'sentence-transformers', 'faiss-cpu', '--quiet'])


# In[4]:


import os
import re
from pdfminer.high_level import extract_text

DOCS_FOLDER = 'sample_docs'
def read_pdf(file_path):
    text = extract_text(file_path)
    if not text:
        return None
    text = text.replace('\f', '\n')          
    text = re.sub(r'[ \t]+', ' ', text)      
    text = re.sub(r'\n{3,}', '\n\n', text)   
    return text.strip()

def read_all_documents(folder):
    documents = {}
    for filename in sorted(os.listdir(folder)):
        if filename.endswith('.pdf'):
            text = read_pdf(os.path.join(folder, filename))
        elif filename.endswith('.txt'):
            with open(os.path.join(folder, filename), 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read().strip()
        else:
            continue
        if text:
            documents[filename] = text
            print(f'  Read: {filename}  ({len(text)} characters)')
    return documents

print(f'Reading documents from: {DOCS_FOLDER}/')
documents = read_all_documents(DOCS_FOLDER)
print(f'Total documents loaded: {len(documents)}')


# In[5]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
KEYWORDS = {'Invoice': ['invoice', 'inv-', 'bill to', 'total amount', 'total due', 'payment due', 'subtotal', 'unit price', 'remittance'],
    'Resume': ['resume', 'curriculum vitae', 'work experience', 'employment', 'education', 'skills', 'references', 'linkedin', 'objective', 'certification'],
    'Utility Bill': ['utility', 'electricity', 'kwh', 'kilowatt', 'meter reading','account number', 'billing period', 'amount due', 'energy usage'],
    'Other': ['agreement', 'contract', 'memorandum', 'memo', 'report','whereas', 'hereby', 'policy', 'parties']}

EXAMPLES = {'Invoice':      'invoice number date company total amount due payment subtotal tax quantity unit price',
    'Resume':       'resume name email phone skills experience education employment linkedin certification',
    'Utility Bill': 'utility electricity kwh meter reading account billing period amount due energy usage',
    'Other':        'agreement contract memorandum report policy parties terms conditions'}

vectorizer = TfidfVectorizer(ngram_range=(1, 2))
example_matrix = vectorizer.fit_transform(list(EXAMPLES.values()))
example_labels = list(EXAMPLES.keys())


def classify_document(text):
    text_lower = text.lower()
    keyword_scores = {}
    for doc_type, words in KEYWORDS.items():
        score = sum(1 for word in words if word in text_lower)
        keyword_scores[doc_type] = score
    doc_vector = vectorizer.transform([text])
    similarities = cosine_similarity(doc_vector, example_matrix)[0]
    tfidf_scores = {label: float(sim) for label, sim in zip(example_labels, similarities)}
    
    max_kw = max(keyword_scores.values()) or 1
    final_scores = {}
    for doc_type in KEYWORDS:
        kw_norm  = keyword_scores[doc_type] / max_kw
        tfidf    = tfidf_scores.get(doc_type, 0)
        final_scores[doc_type] = round(0.7 * kw_norm + 0.3 * tfidf, 4)
    
    best_type = max(final_scores, key=final_scores.get)
    best_score = final_scores[best_type]
    
    total_keywords = sum(keyword_scores.values())
    if total_keywords < 2 and best_score < 0.15:
        return 'Unclassifiable', 0.0
    
    return best_type, round(best_score, 2)


classifications = {}
for filename, text in documents.items():
    doc_type, confidence = classify_document(text)
    classifications[filename] = (doc_type, confidence)
    print(f'  {filename:<30}  ->  {doc_type:<15}  (confidence: {confidence})')


# In[8]:


def find(pattern, text):
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        for group in match.groups():
            if group:
                return group.strip()
    return None

def to_float(value):
    if value is None:
        return None
    return float(value.replace(',', '').replace(' ', ''))
    
def extract_invoice(text):
    invoice_number = (find(r'invoice\s+(?:number|no\.?|#)\s*[:\-]?\s*([A-Z0-9\-]{3,20})', text) or
            find(r'invoice\s*[:#]\s*([A-Z0-9\-]{3,20})', text) or
            find(r'\b(INV[-#]?\d+)\b', text))
    date = (find(r'(\d{4}[-/]\d{2}[-/]\d{2})', text) or
            find(r'([A-Za-z]+\.?\s+\d{1,2},?\s+\d{4})', text) or
            find(r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})', text) )
    company = (find(r'company\s*[:\-]\s*(.+)', text) or
            find(r'bill\s+to\s*[:\-]?\s*(.+)', text) or
            find(r'([A-Z][A-Za-z\s]+(?:Ltd|LLC|Inc|Corp|Co)\.?)', text))
    if company:
        company = company.rstrip('.,').strip()[:60]  # keep it short

    total_raw = (find(r'total\s+(?:amount|due)?\s*[:\$]?\s*(\d[\d,\s]*\.?\d{0,2})', text) or
                find(r'grand\s+total\s*[:\$]?\s*(\d[\d,\s]*\.?\d{0,2})', text))
    if not total_raw:
        all_amounts = re.findall(r'\$(\d{1,3}(?:[,\s]\d{3})*(?:\.\d{2})?)', text)
        if all_amounts:
            total_raw = all_amounts[-1]

    return {'invoice_number': invoice_number,
            'date':           date,
            'company':        company,
            'total_amount':   to_float(total_raw) }

def extract_resume(text):
    email = find(r'([a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,})', text)
    phone_match = re.search(r'(?:\+?1[\s\-.]?)?(?:\(?\d{3}\)?[\s\-.]?)\d{3}[\s\-.]?\d{4}', text)
    phone = re.sub(r'\s+', '-', phone_match.group(0).strip()) if phone_match else None
    name = None
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    for line in lines[:10]:  # only look in first 10 lines
        words = line.split()
        if (2 <= len(words) <= 5
                and '@' not in line
                and not re.search(r'\d', line)
                and re.match(r"^[A-Za-z\s'\-\.]+$", line)):
            name = line
            break
    years = None
    explicit = find(r'(\d+)\+?\s+years?\s+(?:of\s+)?(?:industry\s+)?experience', text)
    if explicit:
        years = int(explicit)
    else:
        spans = re.findall(r'\((\d{4})\s*[-–]\s*(\d{4}|present)\)', text, re.IGNORECASE)
        if spans:
            total = 0
            for start, end in spans:
                end_year = 2025 if end.lower() == 'present' else int(end)
                total += max(0, end_year - int(start))
            years = total if total > 0 else None

    return {'name':             name,
            'email':            email,
            'phone':            phone,
            'experience_years': years}

def extract_utility_bill(text):
    account_number = (find(r'account\s*(?:number|no\.?|#)?\s*[:\-]?\s*([A-Z0-9\-]{5,20})', text) or
                    find(r'acct\s*[:\-]?\s*([A-Z0-9\-]+)', text))

    date = (find(r'(\d{4}[-/]\d{2}[-/]\d{2})', text) or
            find(r'([A-Za-z]+\.?\s+\d{1,2},?\s+\d{4})', text))
    kwh_values = re.findall(r'([\d,]+(?:\.\d+)?)\s*kwh', text, re.IGNORECASE)
    usage_kwh = None
    if kwh_values:
        numbers = [float(v.replace(',', '')) for v in kwh_values]
        usage_kwh = max(numbers)
    amount_raw = ( find(r'amount\s+due\s*[:\$]?\s*(\d[\d,\s]*\.?\d{0,2})', text) or
                    find(r'total\s+due\s*[:\$]?\s*(\d[\d,\s]*\.?\d{0,2})', text) or
                    find(r'balance\s+due\s*[:\$]?\s*(\d[\d,\s]*\.?\d{0,2})', text))

    return {'account_number': account_number,
            'date':           date,
            'usage_kwh':      usage_kwh,
            'amount_due':     to_float(amount_raw)}

def extract_fields(doc_type, text):
    if doc_type == 'Invoice':
        return extract_invoice(text)
    elif doc_type == 'Resume':
        return extract_resume(text)
    elif doc_type == 'Utility Bill':
        return extract_utility_bill(text)
    else:
        return {}  


print('Extracting structured fields...')
print('-' * 50)

results = {}
for filename, text in documents.items():
    doc_type, confidence = classifications[filename]
    fields = extract_fields(doc_type, text)
    results[filename] = {'class': doc_type, 'confidence': confidence, **fields}
    print(f'  {filename}  ->  {doc_type}')
    for key, val in fields.items():
        print(f'      {key}: {val}')
    print()

print('Done!')


# In[9]:


import json

output_path = 'output.json'

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f'Results saved to: {output_path}')
print()
print(json.dumps(results, indent=2))


# In[10]:


from sklearn.feature_extraction.text import TfidfVectorizer as SearchVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def split_into_chunks(text, chunk_size=400, overlap=80):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += chunk_size - overlap
    return chunks


all_chunks = []
chunk_sources = []  

for filename, text in documents.items():
    for chunk in split_into_chunks(text):
        all_chunks.append(chunk)
        chunk_sources.append(filename)
search_vectorizer = SearchVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=20000)
chunk_matrix = search_vectorizer.fit_transform(all_chunks)

print(f'Search index built from {len(all_chunks)} chunks across {len(documents)} documents.')


# In[11]:


def search(query, top_k=5):
    query_vector = search_vectorizer.transform([query])
    scores = cosine_similarity(query_vector, chunk_matrix)[0]
    ranked_indices = np.argsort(scores)[::-1]
    seen_files = {}
    for idx in ranked_indices:
        score = float(scores[idx])
        if score < 0.001:  
            break
        filename = chunk_sources[idx]
        if filename not in seen_files:
            seen_files[filename] = {
                'filename': filename,
                'score':    round(score, 4),
                'excerpt':  all_chunks[idx].replace('\n', ' ').strip()
            }
        if len(seen_files) >= top_k:
            break

    return sorted(seen_files.values(), key=lambda x: -x['score'])

queries = [ 'payments due in January',
            'machine learning experience',
            'high electricity usage',]

for query in queries:
    print(f'Query: "{query}"')
    print('-' * 50)
    for i, result in enumerate(search(query, top_k=3), 1):
        print(f'  [{i}] {result["filename"]}  (score: {result["score"]})')
        print(f'      {result["excerpt"][:150]}...')
    print()


# In[12]:


my_query = 'software engineer with Java skills'
print(f'Query: "{my_query}"')
for i, result in enumerate(search(my_query, top_k=5), 1):
    print(f'  [{i}] {result["filename"]}  (score: {result["score"]})')
    print(f'      {result["excerpt"][:200]}...')


# In[13]:


from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

st_model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = st_model.encode(all_chunks, show_progress_bar=True, normalize_embeddings=True)
embeddings = embeddings.astype('float32')

dim = embeddings.shape[1]
faiss_index = faiss.IndexFlatIP(dim)
faiss_index.add(embeddings)
print(f'FAISS index ready: {faiss_index.ntotal} vectors')

def search_dense(query, top_k=5):
    q_vec = st_model.encode([query], normalize_embeddings=True).astype('float32')
    distances, indices = faiss_index.search(q_vec, top_k * 3)
    seen = {}
    for score, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        filename = chunk_sources[idx]
        if filename not in seen:
            seen[filename] = {'filename': filename,
                                'score':    round(float(score), 4),
                                'excerpt':  all_chunks[idx].replace('\n', ' ').strip()}
    return sorted(seen.values(), key=lambda x: -x['score'])[:top_k]
query = 'Find all documents mentioning payments due in January'
print(f'Query: "{query}"')
for i, r in enumerate(search_dense(query), 1):
    print(f'  [{i}] {r["filename"]}  (score: {r["score"]})')
    print(f'      {r["excerpt"][:200]}...')

