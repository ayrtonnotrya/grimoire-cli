# Grimoire CLI - Architecture

## Overview
Grimoire is a CLI tool designed to manage a digital library of magic books. It automates the summarization (fichamento) of PDFs using Google's Gemini 2.5 Flash model and enables semantic search over these summaries using ChromaDB.

## Core Components

### 1. CLI Interface (`grimoire/cli.py`)
- Built with **Typer** and **Rich**.
- Provides commands for initialization, processing, and searching.
- Handles user input and output formatting.

### 2. Configuration Management (`grimoire/config.py`)
- Uses **TOML** format.
- Stores API keys, directory paths (library root, summaries output), and model settings.
- Default location: `~/.config/grimoire/config.toml`.

### 3. Core Logic (`grimoire/core.py`)
- **Library Parsing**: Reads a text file containing the file tree of the library to identify PDF files.
- **State Management**: Checks if a summary already exists for a given PDF to avoid redundant processing.
- **Gemini Integration**:
    - Uploads PDFs to Gemini API.
    - Sends a specific prompt (`book_summary_prompt.md`) to generate a structured summary.
    - Saves the response as a Markdown file.
- **Indexing**: Orchestrates the embedding and storage of summaries into the vector database.

### 4. Database Layer (`grimoire/db.py`)
- **ChromaDB**: Used as the vector database.
- **Embeddings**: Uses Gemini's embedding model (or a compatible alternative) to vectorize the summaries.
- **Search**: Performs semantic search queries against the indexed summaries.

## Data Flow

1.  **Input**: User provides a library tree file.
2.  **Processing**:
    -   System identifies PDFs.
    -   Checks existence of summary.
    -   If missing -> Upload to Gemini -> Generate Summary(2k tokens max) -> Save MD.
3.  **Indexing**:
    -   System reads generated Markdown summaries.
    -   Generates embeddings for documents.
    -   Stores in ChromaDB.
4.  **Search**:
    -   User queries via CLI.
    -   Query is embedded.
    -   ChromaDB returns relevant summaries.

## Directory Structure
```
grimoire-cli/
├── docs/               # Documentation
├── grimoire/           # Source code
│   ├── __init__.py
│   ├── cli.py          # Entry point
│   ├── config.py       # Config manager
│   ├── core.py         # Business logic
│   └── db.py           # Database wrapper
├── tests/              # Automated tests
├── pyproject.toml      # Dependencies (Poetry)
└── README.md
```
