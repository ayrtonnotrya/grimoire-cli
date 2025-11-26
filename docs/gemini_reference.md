# Gemini API Reference for Grimoire

This document details the usage of the Google Gemini API for the Grimoire project, specifically focusing on the **Gemini 2.5 Flash** model and the **google-genai** SDK.

## 1. Model: Gemini 2.5 Flash
**Code**: `gemini-2.5-flash`
**Why**: Best price-performance, high throughput, low latency. Ideal for processing large libraries.

### Capabilities
-   **Context Window**: 1,048,576 tokens (Input), 65,536 tokens (Output).
-   **Features**: Structured Outputs, JSON Mode, Document Understanding (PDFs), Caching.

> **More details**: [Gemini Models](genai/gemini_models.md)

## 2. PDF Processing (Document Understanding)
Gemini 2.5 Flash can natively understand PDFs (text, images, charts).

> **More details**: [Document Understanding](genai/document_understanding.md)

### Method A: Inline Data (Recommended for < 20MB)
Pass the PDF bytes directly in the request. Best for one-off processing.
```python
from google import genai
from google.genai import types

client = genai.Client(api_key="...")
with open("book.pdf", "rb") as f:
    pdf_data = f.read()

# Load prompt
with open("grimoire/templates/book_summary_prompt.md", "r") as f:
    prompt_text = f.read()

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[
        types.Part.from_bytes(data=pdf_data, mime_type='application/pdf'),
        prompt_text
    ]
)
```

### Method B: File API (Recommended for > 20MB or Reuse)
Upload the file first. Best for very large files or if you plan to query the same PDF multiple times.
-   **Limit**: 50MB or 1000 pages per file.
-   **Retention**: Files stored for 48 hours.
```python
# Upload
file = client.files.upload(file="book.pdf", config={'mime_type': 'application/pdf'})

# Generate
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[file, prompt_text]
)
```

## 3. Structured Outputs (Fichamento Format)
To ensure the summary follows a strict format, use **Structured Outputs** with Pydantic.

> **More details**: [Structured Outputs](genai/structured_outputs.md)

```python
from pydantic import BaseModel

class BookSummary(BaseModel):
    title: str
    author: str
    key_concepts: list[str]
    summary_text: str

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[
        # Assuming 'file' or 'pdf_data' is already prepared as above
        types.Part.from_bytes(data=pdf_data, mime_type='application/pdf'), 
        prompt_text
    ],
    config={
        "response_mime_type": "application/json",
        "response_json_schema": BookSummary.model_json_schema(),
    },
)
# Parse result
summary = BookSummary.model_validate_json(response.text)
```

## 3.1. Limiting Output Length (for Embeddings)

**Problem**: The embedding model (`gemini-embedding-001`) has a **2048 token input limit**. If your summary exceeds this, it cannot be indexed.

**Solution**: Use `max_output_tokens` in `GenerateContentConfig` to constrain the response length.

> **More details**: [Python SDK Reference](genai/genai_python_sdk.md#system-instructions-and-other-configs)

```python
from google.genai import types

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[
        types.Part.from_bytes(data=pdf_data, mime_type='application/pdf'),
        prompt_text
    ],
    config=types.GenerateContentConfig(
        max_output_tokens=2000,  # Ensures summary fits within embedding limit
        response_mime_type="application/json",
        response_json_schema=BookSummary.model_json_schema(),
    ),
)
```

**Best Practice**: Set `max_output_tokens` to ~1500-1800 to leave room for JSON structure overhead.


## 4. Embeddings (Vector Search)
**Model**: `gemini-embedding-001`
**Dimensions**: 3072 (default), but can be set to 768 for efficiency.

> **More details**: [Embeddings](genai/embeddings.md)

### Task Types
Critical for performance.
-   **Indexing**: `task_type="RETRIEVAL_DOCUMENT"` (for the summaries).
-   **Searching**: `task_type="RETRIEVAL_QUERY"` (for the user's query).

```python
# Indexing a summary
result = client.models.embed_content(
    model="gemini-embedding-001",
    contents="Summary text...",
    config={'task_type': 'RETRIEVAL_DOCUMENT'}
)
embedding = result.embeddings[0].values

# Searching
query_result = client.models.embed_content(
    model="gemini-embedding-001",
    contents="Search query...",
    config={'task_type': 'RETRIEVAL_QUERY'}
)
```

## 5. Rate Limits (Gemini 2.5 Flash)
Limits apply per **Project**.

> **More details**: [Rate Limits](genai/rate_limits.md)

| Tier | RPM (Requests/Min) | TPM (Tokens/Min) | RPD (Requests/Day) |
| :--- | :--- | :--- | :--- |
| **Free** | 10 | 250,000 | 250 |
| **Paid (Tier 1)** | 1,000 | 1,000,000 | 10,000 |

**Strategy**: Implement exponential backoff for `429 Too Many Requests` errors.

## 6. SDK Migration Note
**Library**: `google-genai` (v0.2.0+)
**Do NOT use**: `google-generativeai` (Legacy)

The project is currently configured to use `google-genai`.
