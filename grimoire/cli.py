import typer
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from grimoire import core, db
from grimoire.config import config
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
    exclude: str = typer.Option(None, "--exclude", "-e", help="🚫 Path to a file listing PDFs to exclude/ignore"),
    sequential: bool = typer.Option(False, "--sequential", "-s", help="🔄 Process files sequentially (default is random order)"),
    index: bool = typer.Option(True, help="⚡ Auto-index summaries for instant searchability"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="🔍 See the magic happen in real-time"),
    delay: float = typer.Option(6.0, "--delay", "-d", help="⏱️ Minimum delay (seconds) between API calls to avoid burst limits"),
    model: str = typer.Option(None, "--model", "-m", help="Gemini Model to use (overrides config)")
):
    """🎭 Unleash the power of AI! Process entire libraries, extracting deep summaries and insights from every book. Watch as wisdom is distilled and made searchable."""
    console.print(f"Processing library from: {list_file}")
    if exclude:
        console.print(f"Excluding files from: {exclude}")
    core.process_library(list_file, exclude_file_path=exclude, sequential=sequential, verbose=verbose, delay=delay, model_name=model)
    
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
def process_file(
    file_path: str = typer.Argument(..., help="Path to the PDF file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    json_output: bool = typer.Option(False, "--json", help="Output result as JSON")
):
    """Process a single file."""
    path = Path(file_path)
    if not path.exists():
        if json_output:
            print(json.dumps({"status": "error", "message": "File not found"}))
        else:
            console.print(f"[red]File not found: {file_path}[/red]")
        raise typer.Exit(code=1)

    result = core.process_single_file(path, verbose=verbose)
    
    if json_output:
        print(json.dumps(result))
    else:
        core._print_process_result(console, result, verbose=verbose)
    
    if result["status"] == "error":
        raise typer.Exit(code=2)

@app.command()
def index_file(
    summary_path: str = typer.Argument(..., help="Path to the summary JSON file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    json_output: bool = typer.Option(False, "--json", help="Output result as JSON")
):
    """Index a single summary file."""
    path = Path(summary_path)
    if not path.exists():
        if json_output:
            print(json.dumps({"status": "error", "message": "File not found"}))
        else:
            console.print(f"[red]File not found: {summary_path}[/red]")
        raise typer.Exit(code=1)

    result = core.index_single_file(path, verbose=verbose)
    
    if json_output:
        print(json.dumps(result))
    else:
        core._print_index_result(console, result, verbose=verbose)

    if result["status"] == "error":
        raise typer.Exit(code=2)

@app.command()
def get_summary(
    pdf_name: str = typer.Argument(..., help="Name of the PDF (e.g., 'book.pdf' or just 'book')"),
    json_output: bool = typer.Option(True, "--json", help="Output as JSON (default: True)")
):
    """Get the full summary of a book as JSON."""
    summary = core.get_summary_json(pdf_name)
    if summary:
        print(json.dumps(summary, indent=2))
    else:
        if json_output:
            print(json.dumps({"status": "error", "message": "Summary not found"}))
        else:
            console.print(f"[red]Summary not found for: {pdf_name}[/red]")
        raise typer.Exit(code=1)

@app.command()
def ask(
    question: str = typer.Argument(..., help="The question to ask the Grimoire"),
    json_output: bool = typer.Option(False, "--json", help="Output result as JSON")
):
    """Ask a question to the Grimoire (RAG)."""
    answer = core.ask_question(question)
    
    if json_output:
        print(json.dumps({"question": question, "answer": answer}))
    else:
        console.print(f"[bold]Question:[/bold] {question}")
        console.print("[bold green]Answer:[/bold green]")
        if answer:
            console.print(Markdown(answer))
        else:
            console.print("[red]No answer received.[/red]")
            
@app.command()
def commune(
    model: str = typer.Option("gemini-2.5-flash", help="Model to use for conversation"),
    only_prompt: bool = typer.Option(False, "--only-prompt", help="Print the Oracle prompt and skip generation. Useful for external use."),
    min_queries: int = typer.Option(5, "--min-queries", "-mq", help="Minimum number of search queries to generate"),
    results: int = typer.Option(40, "--results", "-r", help="Number of results per query")
):
    """🔮 Commune with the Grimoire. An interactive, RAG-enhanced chat session."""
    core.start_commune_session(
        model_name=model, 
        only_prompt=only_prompt, 
        min_queries=min_queries, 
        results_per_query=results,
        session_type="commune"
    )

@app.command()
def tarot(
    question: str = typer.Argument(None, help="The question for the cards (optional, can be entered interactively)"),
    deck: str = typer.Option("Rider-Waite-Smith", "--deck", "-d", help="The Tarot deck used"),
    spread: str = typer.Option("Celtic Cross", "--spread", "-s", help="The spread or technique used"),
    cards: str = typer.Option(None, "--cards", "-c", help="Comma-separated list of cards drawn (optional)"),
    model: str = typer.Option("gemini-2.5-flash", help="Model to use for conversation"),
    only_prompt: bool = typer.Option(False, "--only-prompt", help="Print the Oracle prompt and skip generation."),
    min_queries: int = typer.Option(5, "--min-queries", "-mq", help="Minimum number of search queries to generate"),
    results: int = typer.Option(40, "--results", "-r", help="Number of results per query")
):
    """🃏 Consult the Tarot through the lens of the Grimoire. Specialized for divination."""
    console.print(Panel.fit(f"[bold magenta]🔮 TAROT READING[/bold magenta]\nDeck: {deck}\nSpread: {spread}", border_style="magenta"))
    
    if question:
        console.print(f"[bold green]Seeker:[/bold green] {question}")
        # Pre-seed history via context or modifies start_commune to take initial input?
        # start_commune_session interacts via input() loop. 
        # For simplicity, we just start the session and let the user re-type or just CopyPaste if they want, 
        # OR we can pass it as initial 'user_input' if we refactored start_commune_session further.
        # But wait, start_commune_session is a loop.
        # Let's just enter the loop. The first "input" is what triggers the first search.
        console.print("[dim]Please re-enter your question to begin the reading session...[/dim]")

    context = {
        "deck": deck,
        "spread": spread,
        "cards": cards or "Not specified (User will provide)"
    }

    core.start_commune_session(
        model_name=model,
        only_prompt=only_prompt,
        min_queries=min_queries,
        results_per_query=results,
        session_type="tarot",
        context_variables=context
    )

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


@app.command()
def report():
    """Generates a statistical report of the library."""
    from grimoire.stats import print_report
    print_report()

@app.command()
def repair(
    list_file: str = typer.Option(..., "--list", "-l", help="📚 Path to the library list file"),
    timeout: int = typer.Option(60, help="⏱️ Timeout in seconds for Ghostscript (default: 60)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="🔍 Verbose output")
):
    """🛠️ Attempt to repair corrupted PDFs using Ghostscript and verify with Gemini."""
    console.print(f"[bold]Starting repair process for list: {list_file}[/bold]")
    core.repair_library(list_file, timeout=timeout, verbose=verbose)

@app.command()
def sigil(
    intent: str = typer.Argument(..., help="The magical intent for the sigil (e.g., 'Overcome creative block')"),
    style: str = typer.Option(..., "--style", "-s", help="The artistic or magical style (e.g., 'Chaos Magic', 'Art Nouveau')"),
    aspect_ratio: str = typer.Option("1:1", "--aspect-ratio", "-ar", help="Aspect ratio for the image (1:1, 16:9, 9:16, 3:4, 4:3)"),
    output: str = typer.Option(..., "--output", "-o", help="Path to save the generated sigil image"),
    nano_banana_pro: bool = typer.Option(False, "--nano-banana-pro", help="Use the advanced Gemini 3 Pro Image model (Nano Banana Pro)")
):
    """🎨 Forge a magical sigil using the Sigil Artificer pipeline (Alchemy -> Materialization)."""
    from grimoire import artificer
    
    console.print(Panel.fit(f"[bold magenta]Sigil Artificer[/bold magenta]\nIntent: {intent}\nStyle: {style}\nModel: {'Nano Banana Pro' if nano_banana_pro else 'Imagen 4 Ultra'}", border_style="magenta"))
    
    artificer.generate_sigil(intent, style, aspect_ratio, output, nano_banana_pro)

@app.command()
def set_imagen_key(
    key: str = typer.Argument(..., help="The API Key for the Satellite Project (Paid)")
):
    """🔑 Set the dedicated API key for Imagen (Satellite Project)."""
    from grimoire.guard import ImagenGuard
    
    console.print(Panel.fit("[bold red]⚠️  WARNING: PAID USAGE ALERT ⚠️[/bold red]\n\nThis key will be used for the 'Satellite Project'.\nIt is subject to strict rate limits to prevent financial loss:\n\n• [bold]1 Request per Minute[/bold]\n• [bold]100 Requests per Day[/bold]\n\n[italic]The key will be stored securely in ~/.local/share/grimoire/imagen_key.secret[/italic]", border_style="red"))
    
    confirm = typer.confirm("Do you understand the risks and wish to proceed?")
    if not confirm:
        console.print("[yellow]Aborted.[/yellow]")
        raise typer.Exit()

    try:
        ImagenGuard.save_key(key)
        console.print(f"[bold green]Imagen Key saved securely![/bold green]")
    except Exception as e:
        console.print(f"[bold red]Failed to save key:[/bold red] {e}")


@app.command()
def ritual(
    intent: str = typer.Argument(..., help="The specific intent of the ritual (e.g., 'Banish anxiety', 'Attract wealth')"),
    inventory: str = typer.Option(None, "--inventory", "-i", help="Comma-separated list of available items"),
    inventory_file: str = typer.Option(None, "--inventory-file", "-f", help="Path to a text file containing the inventory"),
    json_output: bool = typer.Option(False, "--json", help="Output the raw JSON structure")
):
    """🕯️ The Ritualist: Construct a personalized ritual based on your intent and available tools."""
    from grimoire.ritual import perform_ritual_planning
    
    # 1. Resolve Inventory
    final_inventory = "No specific tools available. Use body, voice, and mind."
    
    if inventory:
        final_inventory = inventory
    elif inventory_file:
        path = Path(inventory_file)
        if path.exists():
            final_inventory = path.read_text().strip()
        else:
            console.print(f"[yellow]Warning: Inventory file '{inventory_file}' not found. Proceeding without specific inventory.[/yellow]")

    if not json_output:
        console.print(Panel.fit(f"[bold magenta]The Ritualist[/bold magenta]\nIntent: {intent}\nInventory Source: {'User Input' if inventory else ('File' if inventory_file else 'None')}", border_style="magenta"))
        with console.status("[bold green]Consulting the Library and constructing your ritual...[/bold green]", spinner="moon"):
            try:
                ritual_obj = perform_ritual_planning(intent, final_inventory)
            except Exception as e:
                console.print(f"[bold red]Ritual construction failed:[/bold red] {e}")
                raise typer.Exit(code=1)
    else:
        try:
            ritual_obj = perform_ritual_planning(intent, final_inventory)
        except Exception as e:
            print(json.dumps({"error": str(e)}))
            raise typer.Exit(code=1)

    # 2. Output Handling
    if json_output:
        print(ritual_obj.model_dump_json(indent=2))
    else:
        # Render Beautifully with Rich
        from rich.table import Table
        from rich.text import Text
        from rich.columns import Columns
        
        # Header
        console.print(f"\n[bold underline overline magenta]{ritual_obj.title.upper()}[/bold underline overline magenta]\n")
        
        # Meta Info
        grid = Table.grid(expand=True)
        grid.add_column()
        grid.add_column(justify="right")
        grid.add_row(f"[bold]Intent:[/bold] {ritual_obj.intent}", f"[bold]Timing:[/bold] {ritual_obj.timing}")
        console.print(Panel(grid, border_style="dim"))
        
        # Tools
        tools_table = Table(title="Required Tools", box=None, show_header=True, header_style="bold cyan")
        tools_table.add_column("Tool")
        tools_table.add_column("Usage")
        tools_table.add_column("Substitute", style="italic dim")
        
        for tool in ritual_obj.tools:
            tools_table.add_row(tool.name, tool.usage, tool.substitute or "-")
            
        console.print(Panel(tools_table, border_style="cyan"))
        
        # Steps
        console.print("\n[bold yellow]--- The Ritual Steps ---[/bold yellow]\n")
        
        for i, step in enumerate(ritual_obj.steps, 1):
            step_content = Text()
            step_content.append(f"Action: {step.action}\n", style="white")
            if step.visualization:
                step_content.append(f"Visualize: {step.visualization}\n", style="italic blue")
            if step.incantation:
                step_content.append(f"\nSay: \"{step.incantation}\"", style="bold magenta")
                
            console.print(Panel(step_content, title=f"{i}. {step.name}", border_style="yellow", expand=False))
            
        # Closing & Result
        console.print(Panel(f"[bold]Closing:[/bold] {ritual_obj.closing}\n\n[bold]Expected Result:[/bold] {ritual_obj.expected_result}", border_style="green"))


if __name__ == "__main__":
    app()
