# Grimoire CLI

Grimoire is a powerful CLI tool for managing your digital library of magic books. It automates the creation of structured summaries using Google's Gemini 2.5 Flash model and enables semantic search over your library using ChromaDB.

## Features

-   **Automated Summarization**: Sends PDF content to Gemini and generates detailed, structured summaries based on a custom prompt.
-   **Smart Search**: Uses vector embeddings to allow semantic search across your entire library of summaries.
-   **Library Management**: Parses your library tree to identify new books and skip already processed ones.

## Installation

1.  **Install Dependencies**:
    ```bash
    poetry install
    ```

2.  **Initialize**:
    ```bash
    poetry run grimoire init
    ```
    Follow the prompts to set your Gemini API Key and directories.

## Usage

### Process Your Library
To scan your library and generate summaries for new books:
```bash
poetry run grimoire process --list /path/to/library_tree.txt
```
*Note: The tool looks for `library_tree_*.txt` files if a directory is provided.*

### Search
To find books relevant to a specific topic:
```bash
poetry run grimoire search "chaos magic rituals for beginners"
```

## Configuration
Configuration is stored in `~/.config/grimoire/config.toml`.
