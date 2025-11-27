import os
import concurrent.futures
from pathlib import Path
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich.console import Console
from grimoire.logger import logger
from google import genai
from google.genai import types
from grimoire.config import config
from grimoire.schemas import BookSummary, BOOK_SUMMARY_SCHEMA
import json

console = Console()

def get_latest_library_tree(directory: Path) -> Path:
    """Finds the latest library_tree_*.txt file in the directory."""
    files = list(directory.glob("library_tree_*.txt"))
    if not files:
        raise FileNotFoundError(f"No library_tree_*.txt found in {directory}")
    return max(files, key=os.path.getctime)

def parse_library_list(file_path: Path) -> list[Path]:
    """Parses the library list file and returns a list of PDF paths.
    
    Expected format: One absolute PDF path per line.
    Example:
        /path/to/book1.pdf
        /path/to/book2.pdf
    """
    pdf_paths = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue
            
            # Check if line ends with .pdf
            if line.endswith(".pdf"):
                pdf_path = Path(line)
                pdf_paths.append(pdf_path)
    
    return pdf_paths

def check_summary_exists(pdf_name: str) -> bool:
    """Checks if a summary file already exists for the given PDF."""
    summary_path = config.summaries_dir / f"summary_{pdf_name}.json"
    return summary_path.exists()

def process_library(list_file_path: str, verbose: bool = False):
    """Main processing function."""
    file_path = Path(list_file_path)
    
    # If directory is passed, find latest tree file
    if file_path.is_dir():
        try:
            file_path = get_latest_library_tree(file_path)
            console.print(f"[blue]Using latest library tree: {file_path}[/blue]")
        except FileNotFoundError as e:
            console.print(f"[red]{e}[/red]")
            return

    console.print(f"[bold]Parsing {file_path}...[/bold]")
    pdf_paths = parse_library_list(file_path)
    console.print(f"Found {len(pdf_paths)} PDFs.")

    api_keys = config.gemini_api_keys
    if not api_keys:
        console.print("[red]Error: Gemini API Key not configured. Run 'grimoire init'.[/red]")
        return

    # Filter out PDFs that already have summaries
    pdfs_to_process = []
    skipped_count = 0
    for pdf_path in pdf_paths:
        if check_summary_exists(pdf_path.name):
             skipped_count += 1
             if verbose:
                console.print(f"[yellow]Skipping existing summary for: {pdf_path.name}[/yellow]")
             continue
        pdfs_to_process.append(pdf_path)

    if skipped_count > 0:
        console.print(f"[yellow]Skipped {skipped_count} existing summaries.[/yellow]")

    if not pdfs_to_process:
        console.print("[green]No new books to process.[/green]")
        return

    console.print(f"[bold blue]Processing {len(pdfs_to_process)} books...[/bold blue]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task_id = progress.add_task("Summarizing...", total=len(pdfs_to_process))
        
        # Parallel Processing
        if len(api_keys) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(api_keys)) as executor:
                futures = []
                for i, pdf_path in enumerate(pdfs_to_process):
                    key = api_keys[i % len(api_keys)]
                    futures.append(executor.submit(generate_summary, pdf_path, key))
                
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    progress.advance(task_id)
                    _print_process_result(progress.console, result)
        else:
            # Sequential Processing
            for pdf_path in pdfs_to_process:
                result = generate_summary(pdf_path, api_keys[0])
                progress.advance(task_id)
                _print_process_result(progress.console, result)

def _print_process_result(console, result):
    color = "green" if result["status"] == "success" else "red"
    icon = "✓" if result["status"] == "success" else "✗"
    
    console.print(
        Panel(
            f"[{color}]{icon} {result['file']}[/{color}]\n[dim]{result['message']}[/dim]",
            border_style=color
        )
    )

def generate_summary(pdf_path: Path, api_key: str) -> dict:
    """Generates a summary for the given PDF using Gemini."""
    if not api_key:
        return {"status": "error", "file": pdf_path.name, "message": "Gemini API Key not provided."}

    if api_key in config.exhausted_keys:
        return {"status": "skipped", "file": pdf_path.name, "message": "API Key exhausted (Daily Limit)"}

    client = genai.Client(api_key=api_key)
    
    try:
        with open(pdf_path, "rb") as f:
            pdf_data = f.read()
    except Exception as e:
        return {"status": "error", "file": pdf_path.name, "message": f"Failed to read file: {e}"}

    # Load prompt
    # Try to find it in the package templates directory
    current_dir = Path(__file__).parent
    prompt_path = current_dir / "templates" / "book_summary_prompt.md"
    
    if not prompt_path.exists():
         return {"status": "error", "file": pdf_path.name, "message": f"Prompt file not found at {prompt_path}"}
    
    with open(prompt_path, "r") as f:
        prompt_text = f.read()

    try:
        response = client.models.generate_content(
            model=config.model_name,
            contents=[
                types.Part.from_bytes(
                    data=pdf_data,
                    mime_type='application/pdf',
                ),
                prompt_text
            ],
            config={
                'response_mime_type': 'application/json',
                'response_schema': BOOK_SUMMARY_SCHEMA,
                'safety_settings': [
                    types.SafetySetting(
                        category='HARM_CATEGORY_HATE_SPEECH',
                        threshold='BLOCK_NONE',
                    ),
                    types.SafetySetting(
                        category='HARM_CATEGORY_DANGEROUS_CONTENT',
                        threshold='BLOCK_NONE',
                    ),
                    types.SafetySetting(
                        category='HARM_CATEGORY_SEXUALLY_EXPLICIT',
                        threshold='BLOCK_NONE',
                    ),
                    types.SafetySetting(
                        category='HARM_CATEGORY_HARASSMENT',
                        threshold='BLOCK_NONE',
                    ),
                ]
            }
        )
        
        # Parse JSON
        try:
            summary_text = response.text
            summary_dict = json.loads(summary_text)
            summary_data = BookSummary(**summary_dict)
            
            # Save JSON
            json_path = config.summaries_dir / f"summary_{pdf_path.name}.json"
            config.summaries_dir.mkdir(parents=True, exist_ok=True)
            
            with open(json_path, "w") as f:
                f.write(summary_data.model_dump_json(indent=2))
                
            return {"status": "success", "file": pdf_path.name, "message": f"Saved to {json_path.name}"}

        except Exception as e:
             error_msg = f"Failed to parse/save response: {e}"
             logger.error(error_msg, exc_info=True)
             return {"status": "error", "file": pdf_path.name, "message": error_msg}

    except Exception as e:
        error_msg = f"Generation failed: {e}"
        if "429" in str(e) or "ResourceExhausted" in str(e):
             masked_key = f"...{api_key[-4:]}"
             
             # Check for Daily Limit
             if "RequestsPerDay" in str(e) or "Daily" in str(e):
                 config.exhausted_keys.add(api_key)
                 error_msg = f"Daily Rate Limit Exceeded (Key: {masked_key}) - Key disabled for session"
             else:
                 error_msg = f"Rate Limit Exceeded (Key: {masked_key})"
        
        logger.error(f"Error processing {pdf_path.name}: {e}", exc_info=True)
        return {"status": "error", "file": pdf_path.name, "message": error_msg}



def index_summaries(verbose: bool = False):
    """Reads all summary files and indexes them in ChromaDB."""
    from grimoire import db
    summaries_dir = config.summaries_dir
    if not summaries_dir.exists():
        console.print(f"[yellow]No summaries directory found at {summaries_dir}[/yellow]")
        return

    # Look for JSON files now
    files = list(summaries_dir.glob("summary_*.json"))
    if not files:
        console.print("[yellow]No JSON summaries found to index.[/yellow]")
        return

    documents = []
    metadatas = []
    ids = []

    console.print(f"Found {len(files)} summaries. Checking index status...")

    api_keys = config.gemini_api_keys
    if not api_keys:
         console.print("[red]Error: Gemini API Key not configured.[/red]")
         return

    # Pre-calculate stats
    already_indexed = 0
    to_index = []
    
    for file_path in files:
        if db.document_exists(file_path.name):
            # Check if path needs updating
            stored_path = db.get_document_path(file_path.name)
            full_path = str(file_path.absolute())
            if stored_path != full_path:
                to_index.append(file_path) # Needs update
            else:
                already_indexed += 1
        else:
            to_index.append(file_path)

    console.print(f"Total: {len(files)} | Indexed: {already_indexed} | To Index: {len(to_index)}")
    
    if not to_index:
        console.print("[green]All summaries are already indexed.[/green]")
        return

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task_id = progress.add_task("Indexing...", total=len(to_index))

        if len(api_keys) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(api_keys)) as executor:
                futures = []
                for file_path in to_index:
                    # Pass ALL keys to the worker, let it manage retries
                    futures.append(executor.submit(_index_single_book, file_path, api_keys))
                
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        progress.advance(task_id)
                        progress.refresh() # Force refresh
                        if verbose or result["status"] == "error" or result.get("path_updated"):
                             _print_index_result(progress.console, result)
                    except Exception as e:
                        error_msg = f"Critical worker error: {e}"
                        progress.console.print(f"[red]{error_msg}[/red]")
                        logger.error(error_msg, exc_info=True)
        else:
            for file_path in to_index:
                result = _index_single_book(file_path, api_keys)
                progress.advance(task_id)
                progress.refresh()
                if verbose or result["status"] == "error" or result.get("path_updated"):
                     _print_index_result(progress.console, result)

def _print_index_result(console, result):
    if result["status"] == "skipped":
        return

    color = "green" if result["status"] == "success" else "red"
    if result.get("path_updated"):
        color = "yellow"
        
    icon = "✓"
    if result["status"] == "error":
        icon = "✗"
    elif result.get("path_updated"):
        icon = "↻"

    msg = f"[{color}]{icon} {result['file']}[/{color}]"
    if result.get("chunks"):
        msg += f" | {result['chunks']} chunks"
    if result.get("message"):
        # Highlight specific errors
        error_msg = result['message']
        if "429" in error_msg or "ResourceExhausted" in error_msg:
             base_msg = "[bold red]Rate Limit Exceeded (Retried)[/bold red]"
             if "(Key:" in error_msg:
                 # Extract key part safely
                 try:
                     key_part = error_msg.split("(Key:")[-1].split(")")[0].strip()
                     error_msg = f"{base_msg} [yellow](Key: {key_part})[/yellow]"
                 except:
                     error_msg = base_msg
             else:
                 error_msg = base_msg
        msg += f" | {error_msg}"
        
    console.print(msg)
    
    # Log result to file
    log_msg = f"{result['status'].upper()}: {result['file']}"
    if result.get("message"):
        log_msg += f" - {result['message']}"
    
    if result["status"] == "error":
        logger.error(log_msg)
    else:
        logger.info(log_msg)

def _index_single_book(file_path: Path, api_keys: list[str]) -> dict:
    from grimoire import db
    import random
    
    # Extract PDF name from summary filename: summary_X.json -> X
    pdf_name = file_path.name.replace("summary_", "").replace(".json", "")

    # Filter out exhausted keys
    available_keys = [k for k in api_keys if k not in config.exhausted_keys]
    
    if not available_keys:
        return {"status": "skipped", "file": pdf_name, "message": "All API Keys exhausted (Daily Limit)"}
    
    # Calculate absolute path for the summary file
    full_path = str(file_path.absolute())

    if db.document_exists(file_path.name):
        # Check if path needs updating
        stored_path = db.get_document_path(file_path.name)
        if stored_path != full_path:
            db.update_document_path(file_path.name, full_path)
            return {"status": "success", "file": pdf_name, "path_updated": True, "message": "Path updated"}
        
        return {"status": "skipped", "file": pdf_name}
    
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
            summary = BookSummary(**data)
    except Exception as e:
        return {"status": "error", "file": pdf_name, "message": f"Failed to load: {e}"}

    documents = []
    metadatas = []
    ids = []

    # Common Metadata (Parental)
    base_metadata = {
        "source": pdf_name,
        "filename": file_path.name,
        "full_path": str(file_path.absolute()),
        "title": summary.header.title,
        "authors": ", ".join(summary.header.authors),
        "category": summary.header.category,
        "keywords": ", ".join(summary.header.keywords)
    }

    # Chunk 1: Central Thesis
    documents.append(f"Central Thesis: {summary.central_thesis}")
    metadatas.append({**base_metadata, "chunk_type": "central_thesis"})
    ids.append(f"{pdf_name}_thesis")

    # Chunk 2: Structure/Chapters
    for i, chapter in enumerate(summary.structure_content):
        content = f"Chapter: {chapter.chapter_title}\n{chapter.summary}"
        documents.append(content)
        metadatas.append({**base_metadata, "chunk_type": "chapter", "chapter_title": chapter.chapter_title})
        ids.append(f"{pdf_name}_chapter_{i}")

    # Chunk 3: Key Concepts
    for i, concept in enumerate(summary.key_concepts):
        content = f"Concept: {concept.term}\nDefinition: {concept.definition}"
        documents.append(content)
        metadatas.append({**base_metadata, "chunk_type": "concept", "term": concept.term})
        ids.append(f"{pdf_name}_concept_{i}")

    # Chunk 4: Practical System
    if summary.practical_system:
        content = f"Practical System: {summary.practical_system.description}\n"
        if summary.practical_system.tools:
            content += f"Tools: {', '.join(summary.practical_system.tools)}\n"
        if summary.practical_system.rituals:
            content += f"Rituals: {', '.join(summary.practical_system.rituals)}"
        
        documents.append(content)
        metadatas.append({**base_metadata, "chunk_type": "practical_system"})
        ids.append(f"{pdf_name}_practical")

    # Chunk 5: Critical Analysis
    content = f"Critical Analysis:\nRelevance: {summary.critical_analysis.relevance}\nTarget Audience: {summary.critical_analysis.target_audience}"
    documents.append(content)
    metadatas.append({**base_metadata, "chunk_type": "critical_analysis"})
    ids.append(f"{pdf_name}_analysis")

    try:
        if documents:
            # Retry Loop for Embeddings
            last_error = None
            
            while available_keys:
                # Pick a key (randomly to distribute load)
                current_key = random.choice(available_keys)
                
                try:
                    # Generate embeddings in parallel (outside the lock)
                    embeddings = db.generate_embeddings(documents, current_key)
                    
                    # Add documents with pre-calculated embeddings (inside the lock)
                    db.add_documents(documents, metadatas, ids, embeddings=embeddings)
                    return {"status": "success", "file": pdf_name, "chunks": len(documents)}
                
                except RuntimeError as e:
                    # Check if it was a daily limit error (already handled in db.py adding to exhausted_keys)
                    if "Daily Rate Limit Exceeded" in str(e):
                        # Key is already in exhausted_keys, remove from local available list
                        if current_key in available_keys:
                            available_keys.remove(current_key)
                        last_error = e
                        continue # Try next key
                    else:
                        # Other error, fail immediately
                        raise e
            
            # If we exited the loop, it means we ran out of keys
            return {"status": "skipped", "file": pdf_name, "message": f"All keys exhausted. Last error: {last_error}"}

        else:
            return {"status": "skipped", "file": pdf_name, "message": "No documents to index"}
    except Exception as e:
        return {"status": "error", "file": pdf_name, "message": str(e)}



