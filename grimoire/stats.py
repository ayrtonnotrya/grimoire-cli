import os
from pathlib import Path
from collections import Counter
import json
from grimoire.config import config
from grimoire import db
from grimoire.schemas import BookSummary
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich import box

console = Console()

def get_dir_size(path: Path) -> int:
    """Calculates the total size of a directory."""
    total = 0
    if not path.exists():
        return 0
    for entry in os.scandir(path):
        if entry.is_file():
            total += entry.stat().st_size
        elif entry.is_dir():
            total += get_dir_size(Path(entry.path))
    return total

def format_size(size: int) -> str:
    """Formats bytes into human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"

def generate_library_stats() -> dict:
    """Aggregates statistics from the library."""
    
    stats = {
        "total_books": 0,
        "total_snippets": 0,
        "total_authors": 0,
        "total_concepts": 0,
        "total_quotes": 0,
        "categories": Counter(),
        "keywords": Counter(),
        "storage_summaries": "0 B",
        "storage_db": "0 B",
        "authors_list": set()
    }

    # 1. DB Stats
    try:
        stats["total_snippets"] = db.count_documents()
    except Exception as e:
        console.print(f"[yellow]Warning: Could not connect to DB for stats: {e}[/yellow]")
        stats["total_snippets"] = 0

    # 2. Storage Stats
    stats["storage_summaries"] = format_size(get_dir_size(config.summaries_dir))
    stats["storage_db"] = format_size(get_dir_size(config.db_dir))

    # 3. Summary Analysis
    summaries_dir = config.summaries_dir
    if not summaries_dir.exists():
        return stats

    json_files = list(summaries_dir.glob("summary_*.json"))
    stats["total_books"] = len(json_files)

    for file_path in json_files:
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                summary = BookSummary(**data)
                
                # Authors
                for author in summary.header.authors:
                    stats["authors_list"].add(author.strip())
                
                # Categories
                if summary.header.category:
                    stats["categories"][summary.header.category] += 1
                
                # Keywords
                for keyword in summary.header.keywords:
                    stats["keywords"][keyword.lower().strip()] += 1
                
                # Concepts
                stats["total_concepts"] += len(summary.key_concepts)
                
                # Quotes
                stats["total_quotes"] += len(summary.relevant_quotes)

        except Exception:
            continue

    stats["total_authors"] = len(stats["authors_list"])
    
    # 4. Key Stats
    stats["total_keys"] = len(config.gemini_api_keys)
    stats["masked_keys"] = [f"...{k[-4:]}" if len(k) > 4 else k for k in config.gemini_api_keys]
    
    return stats

def print_report():
    """Prints a formatted report to the console."""
    with console.status("[bold green]Analyzing library...[/bold green]"):
        stats = generate_library_stats()

    # Layout
    grid = Table.grid(expand=True)
    grid.add_column(justify="center", ratio=1)
    grid.add_column(justify="center", ratio=1)

    # General Stats Table
    general_table = Table(title="General Statistics", box=box.ROUNDED)
    general_table.add_column("Metric", style="cyan")
    general_table.add_column("Value", style="magenta")
    
    general_table.add_row("Total Books", str(stats["total_books"]))
    general_table.add_row("Total Snippets (DB)", str(stats["total_snippets"]))
    general_table.add_row("Total Authors", str(stats["total_authors"]))
    general_table.add_row("Total Concepts", str(stats["total_concepts"]))
    general_table.add_row("Total Quotes", str(stats["total_quotes"]))
    general_table.add_row("Storage (Summaries)", stats["storage_summaries"])
    general_table.add_row("Storage (Vector DB)", stats["storage_db"])

    # API Keys Table
    keys_table = Table(title="API Keys", box=box.ROUNDED)
    keys_table.add_column("Metric", style="cyan")
    keys_table.add_column("Value", style="magenta")
    
    keys_table.add_row("Total Keys", str(stats["total_keys"]))
    keys_table.add_row("Keys List", ", ".join(stats["masked_keys"]))

    # Categories Table
    cat_table = Table(title="Top Categories", box=box.ROUNDED)
    cat_table.add_column("Category", style="green")
    cat_table.add_column("Count", style="yellow")
    
    for cat, count in stats["categories"].most_common(10):
        cat_table.add_row(cat, str(count))

    # Keywords Table
    kw_table = Table(title="Top Keywords", box=box.ROUNDED)
    kw_table.add_column("Keyword", style="blue")
    kw_table.add_column("Count", style="yellow")
    
    for kw, count in stats["keywords"].most_common(10):
        kw_table.add_row(kw, str(count))

    # Print
    console.print(Panel("[bold]Grimoire Library Report[/bold]", style="bold white on blue"))
    console.print(general_table)
    console.print(keys_table)
    console.print("\n")
    
    # Side by side for categories and keywords
    side_table = Table.grid(expand=True, padding=(0, 2))
    side_table.add_column(ratio=1)
    side_table.add_column(ratio=1)
    side_table.add_row(cat_table, kw_table)
    
    console.print(side_table)
