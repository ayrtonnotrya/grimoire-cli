import typer
from rich.console import Console
from grimoire.config import config
from grimoire import core, db

app = typer.Typer(help="Grimoire: Magic Book Summarization and Search Tool")
console = Console()

@app.command()
def init():
    """Initialize configuration."""
    console.print("[bold green]Initializing Grimoire...[/bold green]")
    
    api_key = typer.prompt("Enter your Gemini API Key", hide_input=True)
    config.gemini_api_key = api_key
    
    summaries_dir = typer.prompt("Enter directory to save summaries", default="./fichamentos")
    config.summaries_dir = summaries_dir
    
    config.save()
    console.print(f"[bold green]Configuration saved to {config.CONFIG_FILE}[/bold green]")

@app.command()
def process(
    list_file: str = typer.Option(..., "--list", "-l", help="Path to the library tree file"),
    index: bool = typer.Option(True, help="Auto-index after processing")
):
    """Process books from the library list."""
    console.print(f"Processing library from: {list_file}")
    core.process_library(list_file)
    
    if index:
        console.print("[bold]Starting indexing...[/bold]")
        core.index_summaries()

@app.command()
def search(query: str, n: int = typer.Option(5, help="Number of results")):
    """Search the library."""
    console.print(f"Searching for: {query}")
    results = db.query_documents(query, n_results=n)
    
    # Results structure: {'ids': [[]], 'distances': [[]], 'metadatas': [[]], 'documents': [[]]}
    if not results['ids'] or not results['ids'][0]:
        console.print("[yellow]No results found.[/yellow]")
        return

    for i, doc_id in enumerate(results['ids'][0]):
        metadata = results['metadatas'][0][i]
        distance = results['distances'][0][i] if results['distances'] else 0.0
        document = results['documents'][0][i]
        chunk_type = metadata.get('chunk_type', 'Unknown').replace('_', ' ').title()
        
        console.print(f"\n[bold blue]{i+1}. {metadata.get('title', 'Unknown Title')}[/bold blue] (Score: {distance:.4f})")
        console.print(f"[cyan]{chunk_type}[/cyan] | File: {metadata.get('filename')}")
        console.print(f"[italic]{document[:300]}...[/italic]")


if __name__ == "__main__":
    app()
