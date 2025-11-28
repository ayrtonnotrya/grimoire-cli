import typer
from rich.console import Console
from grimoire.config import config
from grimoire import core, db
import json
from pathlib import Path

app = typer.Typer(help="✨ Grimoire: Transform your entire library into a searchable oracle of knowledge! Harness AI to distill wisdom from countless books and discover insights instantly. 🔮")
console = Console()

@app.command()
def init():
    """🌟 Begin your magical journey! Set up Grimoire with your API key and unlock the power to transform any library into an intelligent, searchable knowledge base."""
    console.print("[bold green]Initializing Grimoire...[/bold green]")
    
    api_key = typer.prompt("Enter your Gemini API Key", hide_input=True)
    config.gemini_api_key = api_key
    
    summaries_dir = typer.prompt("Enter directory to save summaries", default="./summaries")
    # Resolve to absolute path immediately
    abs_summaries_dir = str(Path(summaries_dir).expanduser().resolve())
    config.summaries_dir = abs_summaries_dir
    
    # Ensure DB parent directory exists
    config.db_dir.parent.mkdir(parents=True, exist_ok=True)
    
    config.save()
    console.print(f"[bold green]Configuration saved to {config.CONFIG_FILE}[/bold green]")

@app.command()
def process(
    list_file: str = typer.Option(..., "--list", "-l", help="📚 Path to your library tree - the gateway to infinite knowledge"),
    index: bool = typer.Option(True, help="⚡ Auto-index summaries for instant searchability"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="🔍 See the magic happen in real-time")
):
    """🎭 Unleash the power of AI! Process entire libraries, extracting deep summaries and insights from every book. Watch as wisdom is distilled and made searchable."""
    console.print(f"Processing library from: {list_file}")
    core.process_library(list_file, verbose=verbose)
    
    if index:
        console.print("[bold]Starting indexing...[/bold]")
        core.index_summaries(verbose=verbose)

@app.command()
def index(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="🔍 Watch the indexing magic unfold")
):
    """🗂️ Transform summaries into a lightning-fast semantic search engine! Build a vector database that understands meaning, not just keywords."""
    console.print("[bold]Starting indexing...[/bold]")
    core.index_summaries(verbose=verbose)

@app.command()
def search(
    query: str, 
    n: int = typer.Option(12, help="🎯 How many gems of wisdom to uncover (default: 12)"),
    json_output: bool = typer.Option(False, "--json", help="📋 Machine-readable oracle output")
):
    """🔮 Ask and you shall receive! Search across your entire library with semantic understanding. Discover hidden connections and profound insights from thousands of pages in seconds."""
    if not json_output:
        console.print(f"Searching for: {query}")
        
    try:
        results = db.query_documents(query, n_results=n)
        
        # Results structure: {'ids': [[]], 'distances': [[]], 'metadatas': [[]], 'documents': [[]]}
        if not results['ids'] or not results['ids'][0]:
            if json_output:
                print("[]")
            else:
                console.print("[yellow]No results found.[/yellow]")
            return

        output_data = []

        for i, doc_id in enumerate(results['ids'][0]):
            metadata = results['metadatas'][0][i]
            distance = results['distances'][0][i] if results['distances'] else 0.0
            document = results['documents'][0][i]
            chunk_type = metadata.get('chunk_type', 'Unknown').replace('_', ' ').title()
            
            # Use full_path from metadata if available, otherwise fallback to constructing it
            stored_path = metadata.get('full_path')
            if stored_path:
                full_path = Path(stored_path)
            else:
                full_path = config.summaries_dir / metadata.get('filename')

            
            if json_output:
                output_data.append({
                    "title": metadata.get('title', 'Unknown Title'),
                    "score": distance,
                    "chunk_type": chunk_type,
                    "filename": metadata.get('filename'),
                    "full_path": str(full_path.absolute()),
                    "snippet": document
                })
            else:
                console.print(f"\n[bold blue]{i+1}. {metadata.get('title', 'Unknown Title')}[/bold blue] (Score: {distance:.4f})")
                console.print(f"[cyan]{chunk_type}[/cyan]")
                console.print(f"Full Summary: {full_path.absolute()}")
                console.print(f"[italic]{document}[/italic]")

        if json_output:
            print(json.dumps(output_data, indent=2))
    except Exception as e:
        console.print(f"[bold red]Search failed:[/bold red] {e}")
        console.print(f"[dim]Check logs for details.[/dim]")
        if json_output:
            print("[]")
        else:
            console.print("[yellow]No results found.[/yellow]")
        return


@app.command()
def deduplicate():
    """🧹 Purify your knowledge base! Remove duplicate entries and keep your library pristine and efficient."""
    count = db.remove_duplicates()
    if count > 0:
        console.print(f"[bold green]Removed {count} duplicate documents.[/bold green]")
    else:
        console.print("[green]No duplicates found.[/green]")


if __name__ == "__main__":
    app()
