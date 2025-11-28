# Grimoire CLI

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**Grimoire** is a powerful CLI tool that transforms your digital library of magic books into a searchable, interactive oracle. It automates the creation of structured summaries using Google's Gemini models and enables semantic search and RAG (Retrieval-Augmented Generation) across your entire collection.

## ✨ Features

-   **Automated Summarization**: Generates detailed, structured summaries for PDF books using Gemini 2.5 Flash.
-   **RAG "Ask" Capability**: Ask complex questions to your library and get context-aware answers based on up to 1000 relevant snippets.
-   **Semantic Search**: Find books and passages by meaning, not just keywords, using ChromaDB vector embeddings.
-   **Smart Key Management**: Supports multiple API keys with automatic rotation and intelligent rate limiting (TPM/RPM) to maximize throughput.
-   **Granular Control**: Process individual files, index specific documents, or scan entire libraries.
-   **JSON Output**: All commands support JSON output for easy integration with other tools or agents.

## 🚀 Installation

### Option 1: Using pipx (Recommended)

Run Grimoire globally without managing dependencies manually:

```bash
# Install directly from GitHub
pipx install git+https://github.com/ayrtonnotrya/grimoire-cli.git

# Initialize configuration
grimoire init
```

### Option 2: Using Poetry (For Development)

1.  **Install Dependencies**:
    ```bash
    poetry install
    ```

2.  **Initialize**:
    ```bash
    poetry run grimoire init
    ```

## 📖 Usage

### 1. Initialize
Set up your API keys and directories:
```bash
grimoire init
```

### 2. Process Your Library
Scan a directory or file list to generate summaries for new books:
```bash
# Process from a file list
grimoire process --list /path/to/library_tree.txt

# Process a single file
grimoire process-file /path/to/book.pdf
```

### 3. Index Summaries
Embed generated summaries into the vector database for searching:
```bash
# Index all processed summaries
grimoire index

# Index a specific summary file
grimoire index-file /path/to/summary.json
```

### 4. Ask the Grimoire (RAG)
Ask questions to your library. The system retrieves relevant context and synthesizes an answer.
```bash
grimoire ask "What are the core principles of Chaos Magic?"
```

### 5. Search
Find books relevant to a specific topic:
```bash
grimoire search "sigil magic rituals" --n 5
```

### 6. Get Summary
Retrieve the structured summary of a specific book:
```bash
grimoire get-summary "Liber Null"
```

## ⚙️ Configuration

Configuration is stored in `~/.config/grimoire/config.toml`.

### Key Management
You can provide multiple Gemini API keys separated by `|` to distribute load and avoid rate limits.
```toml
[gemini]
api_key = "KEY_1|KEY_2|KEY_3"
model_name = "gemini-2.5-flash"
```

### Rate Limits
Grimoire automatically handles rate limits for various Gemini models. You can override them if needed:
```toml
[rate_limits]
rpm = 15
tpm = 1000000
```

## 🛠️ Development

To run tests:
```bash
poetry run pytest
```
