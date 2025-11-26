# Grimoire CLI - Setup Guide

## Prerequisites
- **Python 3.10+**
- **Poetry** (Python dependency manager)
- **Gemini API Key** (Get one from Google AI Studio)

## Installation

1.  **Clone the repository** (if applicable) or navigate to the project directory.

2.  **Install Dependencies**:
    ```bash
    poetry install
    ```

3.  **Initialize Configuration**:
    Run the init command to set up your API key and directories.
    ```bash
    poetry run grimoire init
    ```
    You will be prompted for:
    -   Gemini API Key
    -   Path to your library root (optional)
    -   Path to save summaries (default: `./summaries`)

## Usage

### Processing Books
To process your library and generate summaries:
```bash
poetry run grimoire process --list path/to/library_tree.txt
```
This command will:
1.  Read the file tree.
2.  Identify PDFs.
3.  Generate summaries for new books using Gemini.
4.  Save them to your configured summaries directory.

### Searching
To search your library of summaries:
```bash
poetry run grimoire search "your search query"
```
This will return the most relevant books and snippets based on your query.

## Development
-   **Run Tests**: `poetry run pytest`
-   **Linting**: `poetry run ruff check .`
